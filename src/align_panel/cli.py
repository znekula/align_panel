from aperture.cli import launch

from .align_workflow import build_workflow


def main():
    launch(target=build_workflow)
