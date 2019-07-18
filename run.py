#!/usr/bin/python3
import click
import os
from functools import partial
import vae.config
from vae.load.generic import load_dataset
from vae.utils.round import round_8
from vae.metrics.recall import recall
from vae.metrics.ndcg import ndcg


@click.command()
@click.option('--train', is_flag=True, help='Run training of VAE')
@click.option('--test', is_flag=True, help='Run validation of VAE')
@click.option('--benchmark', is_flag=True, help='Run benchmark of VAE')
@click.option('--use_tf_amp', is_flag=True,
              help='Enable Automatic Mixed Precision to speedup fp32 computation using tensor cores')
@click.option('--dataset', default=vae.config.ML_20M, help='Dataset to use')
@click.option('--gpu_number', default=0, help='Number of GPU used in training or validation')
@click.option('--number_of_gpus', default=1, help='How many GPUs to use during training or validation')
@click.option('--number_of_epochs', default=200, help='Number of epochs to train')
@click.option('--batch_size_train', default=10000)
@click.option('--batch_size_validation', default=10000, help='Used both for validation and testing')
@click.option('--validation_step', default=5)
@click.option('--warm_up_epochs', default=5, help='Number of epochs to omit during benchmark')
@click.option('--total_anneal_steps', default=200000, help='Number of annealing steps')
@click.option('--anneal_cap', default=0.2, help='Annealing cap')
@click.option('--lam', default=1e-2, help='Regularization parameter')
@click.option('--lr', default=1e-3, help='Learning rate')
def main(train,
         test,
         benchmark,
         dataset,
         gpu_number,
         number_of_gpus,
         number_of_epochs,
         batch_size_train,
         batch_size_validation,
         use_tf_amp,
         validation_step,
         warm_up_epochs,
         total_anneal_steps,
         anneal_cap,
         lam,
         lr):
    if not train and not test and not benchmark:
        print("Choose one or more:")
        for option in "train", "test", "benchmark":
            print("\t" + option)
        exit(1)

    if train and benchmark:
        print("Use of train and benchmark together is not allowed")
        exit(1)

    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # set AMP
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' if use_tf_amp else '0'

    # load dataset
    (train_data,
     validation_data_input,
     validation_data_true,
     test_data_input,
     test_data_true) = load_dataset(dataset,
                                    val_ratio=0.0731651997,
                                    test_ratio=0.0731651997,
                                    min_items=5,
                                    min_users=0,
                                    seed=0)

    # make sure all dims and sizes are divisible by 8
    number_of_train_users, number_of_items = train_data.shape
    number_of_items = round_8(number_of_items)

    for data in [train_data,
                 validation_data_input,
                 validation_data_true,
                 test_data_input,
                 test_data_true]:
        number_of_users, _ = data.shape
        data.resize(number_of_users, number_of_items)

    # compute VAE dims
    number_of_users, number_of_items = train_data.shape
    encoder_dims = [number_of_items, 600, 200]

    # create VAE
    if number_of_gpus == 0:
        from vae.models.Mult_VAE_training import VAE
        vae = VAE(train_data,
                  encoder_dims,
                  total_anneal_steps=total_anneal_steps,
                  anneal_cap=anneal_cap,
                  batch_size_train=batch_size_train,
                  batch_size_validation=batch_size_validation,
                  lam=lam,
                  lr=lr,
                  device='/device:CPU')
    elif number_of_gpus == 1:
        from vae.models.Mult_VAE_training import VAE
        vae = VAE(train_data,
                  encoder_dims,
                  total_anneal_steps=total_anneal_steps,
                  anneal_cap=anneal_cap,
                  batch_size_train=batch_size_train,
                  batch_size_validation=batch_size_validation,
                  lam=lam,
                  lr=lr,
                  device='/device:GPU:' + str(gpu_number))
    else:
        batches_per_epoch = -(-number_of_train_users // batch_size_train)  # ceil div
        if (batches_per_epoch % number_of_gpus != 0):
            print("Number of batches must be divisible by number of GPUs")
            exit(1)
        from vae.models.Mult_VAE_training_horovod import VAE_Horovod
        vae = VAE_Horovod(train_data,
                          encoder_dims,
                          total_anneal_steps=total_anneal_steps,
                          anneal_cap=anneal_cap,
                          batch_size_train=batch_size_train,
                          batch_size_validation=batch_size_validation,
                          lam=lam,
                          lr=lr)

    metrics = {'ndcg@100': partial(ndcg, R=100),
               'recall@20': partial(recall, R=20),
               'recall@50': partial(recall, R=50)}

    if train:
        vae.train(n_epochs=number_of_epochs,
                  train_data=train_data,
                  validation_data_input=test_data_input,
                  validation_data_true=test_data_true,
                  batch_size_train=batch_size_train,
                  batch_size_validation=batch_size_validation,
                  metrics=metrics,
                  validation_step=validation_step)
        if number_of_gpus > 1:
            print("Saving is not supported with horovod multigpu yet")
        else:
            vae.save()

    if benchmark:
        vae.benchmark(n_epochs=number_of_epochs,
                      warm_up_epochs=warm_up_epochs,
                      train_data=train_data,
                      validation_data_input=test_data_input,
                      validation_data_true=test_data_true,
                      batch_size_train=batch_size_train,
                      batch_size_validation=batch_size_validation,
                      metrics=metrics,
                      validation_step=validation_step)
        vae.save()

    if test and number_of_gpus <= 1:
        vae.load()

        test_results = \
            vae.test(test_data_input=test_data_input,
                     test_data_true=test_data_true,
                     metrics=metrics)

        for k, v in test_results.items():
            print("{}:\t{}".format(k, v))
    elif test and number_of_gpus > 1:
        print("Testing is not supported with horovod multigpu yet")


if __name__ == '__main__':
    main()
