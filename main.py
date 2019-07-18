import click
import sys
import subprocess
import vae.config


@click.command()
@click.option('--train', is_flag=True, help='Run training of VAE')
@click.option('--test', is_flag=True, help='Run validation of VAE')
@click.option('--benchmark', is_flag=True, help='Run benchmark of VAE')
@click.option('--use_tf_amp', is_flag=True,
              help='Enable Automatic Mixed Precision to speedup fp32 computation using tensor cores')
@click.option('--dataset', default=vae.config.ML_20M, help='Dataset to use')
@click.option('--gpu_number', default=0, help='Number of GPU used in training or validation')
@click.option('--number_of_gpus', default=1,
              help='How many GPUs to use during training or validation')  # TODO doesn't work with saving
@click.option('--number_of_epochs', default=200, help='Number of epochs to train')
@click.option('--batch_size_train', default=10000)
@click.option('--batch_size_validation', default=10000, help='Used both for validation and testing')
@click.option('--validation_step', default=5)
@click.option('--warm_up_epochs', default=5, help='Number of epochs to omit during benchmark')
@click.option('--total_anneal_steps', default=200000, help='Number of annealing steps')
@click.option('--anneal_cap', default=0.2, help='Annealing cap')
@click.option('--lam', default=1e-2, help='Regularization parameter')
@click.option('--lr', default=1e-3, help='Learning rate')
def dispatcher(*args, **kwargs):
    if kwargs['number_of_gpus'] <= 1:
        subprocess.call(['python3', 'run.py'] + sys.argv[1:])
    else:
        subprocess.call(['horovodrun',
                         '-np', str(kwargs['number_of_gpus']),
                         '-H', 'localhost:' + str(kwargs['number_of_gpus']),
                         'python3', 'run.py'] + sys.argv[1:])


if __name__ == '__main__':
    dispatcher()
