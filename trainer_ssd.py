import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import augmentation
from ssd_loss import CustomLoss
from utils import bbox_utils, data_utils, io_utils, train_utils
import ray
from ray.util.sgd.tf.tf_trainer import TFTrainer


def dataset_creator():
    hyper_params = train_utils.get_hyper_params(backbone)
    train_data, info = data_utils.get_dataset("voc/2007", "train+validation")
    val_data, _ = data_utils.get_dataset("voc/2007", "test")
    
    if opt.with_voc12:
        voc_2012_data, voc_2012_info = data_utils.get_dataset("voc/2012", "train+validation")
        voc_2012_total_items = data_utils.get_total_item_size(voc_2012_info, "train+validation")
        train_data = train_data.concatenate(voc_2012_data)
    
    labels = data_utils.get_labels(info)
    labels = ["bg"] + labels
    hyper_params["total_labels"] = len(labels)
    img_size = hyper_params["img_size"]
    
    train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, augmentation.apply))
    val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
    
    data_shapes = data_utils.get_data_shapes()
    padding_values = data_utils.get_padding_values()
    train_data = train_data.shuffle(opt.batch_size*4).padded_batch(opt.batch_size, padded_shapes=data_shapes, padding_values=padding_values)
    val_data = val_data.padded_batch(opt.batch_size, padded_shapes=data_shapes, padding_values=padding_values)

    prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
    ssd_train_feed = train_utils.generator(train_data, prior_boxes, hyper_params)
    ssd_val_feed = train_utils.generator(val_data, prior_boxes, hyper_params)

    return ssd_train_feed, ssd_val_feed


def model_creator():
    ssd_model = get_model(hyper_params)
    ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
    ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
    init_model(ssd_model)
    ssd_model_path = io_utils.get_model_path(backbone)
    if opt.load_weights:
        ssd_model.load_weights(ssd_model_path)
    return ssd_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TF2.SSD with Ray")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--load-weights", default=False, action="store_true")
    parser.add_argument("--with_voc12", default=True, action="store_true")
    parser.add_argument("--backbone", default="vgg16", type=str)
    parser.add_argument("--address",required=False,type=str,help="the address to use for Ray")
    parser.add_argument("--num-replicas","-n",type=int,default=1,help="Sets number of replicas for training.")
    parser.add_argument("--use-gpu",action="store_true",default=False,help="Enables GPU training")
    parser.add_argument("--smoke-test",action="store_true",default=False,help="Finish quickly for testing. Assume False for users.")

    opt = parser.parse_args()
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()
    io_utils.is_valid_backbone(opt.backbone)
    if opt.backbone == "mobilenet_v2":
        from models.ssd_mobilenet_v2 import get_model, init_model
    else:
        from models.ssd_vgg16 import get_model, init_model
    ssd_log_path = io_utils.get_log_path(backbone)

    train_total_items = data_utils.get_total_item_size(info, "train+validation")
    val_total_items = data_utils.get_total_item_size(info, "test")
    if opt.with_voc12:
        train_total_items += voc_2012_total_items

    step_size_train = train_utils.get_step_size(train_total_items, opt.batch_size)
    step_size_val = train_utils.get_step_size(val_total_items, opt.batch_size)

    num_train_steps = 10 if args.smoke_test else step_size_train
    num_eval_steps = 10 if args.smoke_test else step_size_val

    trainer = TFTrainer(
        model_creator=model_creator,
        data_creator=dataset_creator,
        num_replicas=opt.num_replicas,
        use_gpu=args.use_gpu,
        verbose=True,
        config={
            "batch_size": batch_size,
            "fit_config": {
                "steps_per_epoch": num_train_steps,
            },
            "evaluate_config": {
                "steps": num_eval_steps,
            }
        }
    )

    checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
    tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
    learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)

    training_start = time.time()
    num_epochs = 1 if opt.smoke_test else opt.epochs
    for i in range(num_epochs):
        # Train num epochs
        train_stats1 = trainer.train()
        # train_stats1.update(train.validate())
        print(f"iter {i}:", train_stats1)

    dt = (time.time() - training_start)/num_epochs
    print(f"Training on workers takes: {dt:.3f} seconds/epoch")

    model = trainer.get_model()
    trainer.shutdown()

    training_start = time.time()
    ssd_model.fit(
        ssd_train_feed,
        steps_per_epoch=step_size_train,
        validation_data=ssd_val_feed,
        validation_steps=step_size_val,
        epochs=opt.epochs,
        callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback]
    )

    dt = (time.time() - training_start)
    print(f"Training on workers takes: {dt:.3f} seconds/epoch")
