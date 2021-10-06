from datetime import datetime
from time import sleep

import click
import zarr
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from humanize import naturaldelta

from .simulation_functions import prepare_simulation
from .utils import zarr2mrcs, json2star

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
        'job_extra': ['--gres=gpu:1'],
        'walltime': '00:30:00',
        'extra': ["--lifetime", "15m", "--lifetime-stagger", "1m"],
    }

    # create a cluster, connect to it and scale
    cluster = SLURMCluster(**SCARF_GPU_CONFIG)
    client = Client(cluster)

    cluster.adapt(minimum_jobs=0, maximum_jobs=n_gpus)

    # create simulation
    simulation = prepare_simulation(
        input_directory=input_directory,
        output_basename=output_basename,
        n_images=n_images,
        image_sidelength=image_sidelength,
        defocus_range=(min_defocus, max_defocus),
        random_seed=random_seed
    )

    click.echo('\n')
    click.echo('### SPSIM 0.0.1 ###')
    click.echo('killing this process will terminate your simulation')
    click.echo('run spsim from tmux/screen in case of connection instability\n')

    click.echo('to track cluster usage, forward port 8787 (e.g for ui4.scarf.rl.ac.uk)')
    click.echo(f'ssh -N -L 8787:ui4.scarf.rl.ac.uk:8787 <SCARF_USER>@ui4.scarf.rl.ac.uk')
    click.echo('then navigate to...')
    click.echo(f'http://localhost:8787/')
    click.echo('on your local machine\n')

    n_structure_files = len(simulation.config.structure_files)
    click.echo(f'simulating {n_images} images from {n_structure_files} structure files')
    click.echo(f'simulation will request short term use of {n_gpus} GPUs using SLURM')
    click.echo(f'job walltimes are short, your jobs will not block others for long!')
    start_time = datetime.now()
    click.echo(f'started computations at {start_time.strftime("%m/%d/%Y, %H:%M:%S")}')

    jf = f'{simulation.config.output_basename}.json'
    with open(jf, 'w') as f:
        f.write(simulation.json())
    click.echo(f"simulation params stored in '{jf}'")

    zf = simulation.zarr_filename
    click.echo(f"results stored in '{zf}'\n")

    click.echo(f'submitting computations to the cluster takes time')
    click.echo(f'once all jobs are submitted, status of simulation will be printed to the console')
    click.echo(f'\n')

    simulation.execute(client)

    za = zarr.convenience.open(zf)

    while za.nchunks_initialized < za.nchunks:
        now = datetime.now()
        particles_simulated = za.nchunks_initialized
        particles_simulated_str = f'{particles_simulated} / {n_images} particles simulated'
        elapsed_time = naturaldelta(now - start_time, minimum_unit='seconds')

        if particles_simulated > 0:
            time_per_particle = (now - start_time).total_seconds() / particles_simulated
        else:
            time_per_particle = 9999.99
        click.echo(
            f'{particles_simulated_str} in {elapsed_time} (avg. {time_per_particle:.2f}s per particle)        \r',
            nl=False
        )
        sleep(0.1)
    click.echo(f'done!')


@click.command()
@click.option(
    '--input-zarr-file',
    type=click.Path(exists=True),
    prompt=True,
    help='zarr file containing particles',
)
@click.option(
    '--output-mrcs-file',
    type=click.Path(exists=False),
    prompt=True,
    help='output mrcs file'
)
def zarr2mrcs_cli(input_zarr_file, mrcs_file):
    zarr2mrcs(input_zarr_file, mrcs_file)
    return


@click.command()
@click.option(
    '--input-json-file',
    type=click.Path(exists=True),
    prompt=True,
    help='json file containing simulation metadata',
)
@click.option(
    '--output-star-file',
    type=click.Path(exists=False),
    prompt=True,
    help='output mrcs file'
)
def json2star_cli(input_json_file, output_star_file):
    json2star(input_json_file, output_star_file)
    return