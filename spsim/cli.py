import click
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from .functions import prepare_simulation

@click.command()
@click.option(
    '--input-directory',
    type=click.Path(exists=True),
    prompt=True,
    help='input directory containing structure files'
)
@click.option(
    '--output-basename',
    type=str,
    prompt=True,
    help='basename for output files from simulation'
)
@click.option(
    '--n-images',
    type=int,
    prompt=True,
    help='number of images to simulate'
)
@click.option(
    '--image-sidelength',
    type=int,
    prompt=True,
    help='sidelength of simulated images, must be divisible by two'
)
@click.option(
    '--min-defocus',
    type=float,
    prompt=True,
    help='minimum defocus value in microns, positive is underfocus'

)
@click.option(
    '--max-defocus',
    type=float,
    prompt=True,
    help='maximum defocus value in microns, positive is underfocus'
)
@click.option(
    '--random-seed',
    default=None,
    type=int,
    prompt=True,
    help='random seed for reproducing identical simulations'
)
@click.option(
    '--n-gpus',
    default=1,
    type=int,
    prompt=True,
    help='number of gpus to request for this simulation'
)
def spsim_scarf(
        input_directory,
        output_basename,
        n_images,
        image_sidelength,
        min_defocus,
        max_defocus,
        random_seed,
        n_gpus,
):
    # prepare computational resources
    SCARF_GPU_CONFIG = {
        'queue': 'gpu',
        'cores': 1,
        'memory': '32GB',
        'job_extra': ['--gres=gpu:1']
    }

    # create a cluster, connect to it and scale
    cluster = SLURMCluster(**SCARF_GPU_CONFIG)
    client = Client(cluster)

    cluster.scale(jobs=n_gpus)

    # create simulation
    simulation = prepare_simulation(
        input_directory=input_directory,
        output_basename=output_basename,
        n_images=n_images,
        image_sidelength=image_sidelength,
        defocus_range=(min_defocus, max_defocus),
        random_seed=random_seed
    )

    click.echo(f'found {len(simulation.config.structure_files)} structure files')
    click.echo(f'simulating {n_images} images')
    click.echo(f'simulation will request {n_gpus} using SLURM')
    click.echo(f'executing simulation...')

    click.echo('to track progress, forward port 8787 (e.g for ui4.scarf.rl.ac.uk')
    click.echo(f'ssh -N -L 8787:ui4.scarf.rl.ac.uk:8787 <SCARF_USER>@ui4.scarf.rl.ac.uk')
    click.echo('then navigate to...')
    click.echo(f'http://localhost:8787/')
    click.echo('on your local machine')

    particle_stack = simulation.execute()
    click.echo(f'done!')

