import sys

from data_masker.cli import main as cli_main


if __name__ == "__main__":
    if "--web" in sys.argv:
        sys.argv.remove("--web")
        from data_masker.gradio_app import main as gradio_main

        gradio_main()
    else:
        cli_main()
