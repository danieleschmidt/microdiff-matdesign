"""Command line interface for MicroDiff-MatDesign."""

import click
from .core import MicrostructureDiffusion
from .imaging import MicroCTProcessor


@click.group()
@click.version_option()
def main():
    """MicroDiff-MatDesign: AI-driven material optimization."""
    pass


@main.command()
@click.argument('image_path')
@click.option('--alloy', default='Ti-6Al-4V', help='Target alloy')
@click.option('--process', default='laser_powder_bed_fusion', help='Manufacturing process')
@click.option('--samples', default=10, help='Number of parameter samples')
def inverse_design(image_path, alloy, process, samples):
    """Generate process parameters from microstructure image."""
    click.echo(f"Loading microstructure from {image_path}...")
    
    processor = MicroCTProcessor()
    microstructure = processor.load_image(image_path)
    
    model = MicrostructureDiffusion(alloy=alloy, process=process)
    parameters = model.inverse_design(microstructure, num_samples=samples)
    
    click.echo("\nOptimal parameters:")
    for param, value in parameters.items():
        click.echo(f"  {param}: {value}")


if __name__ == '__main__':
    main()