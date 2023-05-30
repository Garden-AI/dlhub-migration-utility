#  Given a DLHub Servable as input, register the model it represents to Garden
#  Isaac Darling
#  May 2023

import os
import requests
from datetime import datetime
from threading import Thread
from itertools import cycle
from time import sleep
from dlhub_sdk import DLHubClient
from garden_ai import GardenClient, Model, step
from garden_ai.mlmodel import LocalModel


dl = DLHubClient()


class Loading:
    """Self contained class for multithreading a loading message during execution

    Args:
        msg (str): Message to print while loading
        complete (str): Message to print when finished
    """
    def __init__(self, msg: str = "Loading...", complete: str = "Finished!") -> None:
        self.msg = f"[ ] {msg}"
        self.complete = f"[*] {complete}"
        self.spinner = ["|", "/", "-", "\\"]
        self.done = False
        self._thread = Thread(target=self._spin, daemon=True)

    def __enter__(self) -> None:
        self.begin()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.done = True

    def begin(self) -> None:
        self._thread.start()

    def _spin(self) -> None:
        for x in cycle(self.spinner):
            if self.done:
                print(f"\r{' ' * len(self.msg)}  \r{self.complete}")
                break
            print(f"\r{self.msg} {x}", end="")
            sleep(0.1)


def get_dlhub_metadata(name: str) -> dict[str, str]:
    """Retrieve the metadata for the servable owned by the caller with the given name

    Args:
        name (str): The name of the servable
    Return:
        (dict): The metadata of the named servable
    """
    return dl.search(f"dlhub.shorthand_name: lsschultz_wisc/{name}", advanced=True, only_latest=True)[0]


def register_model(metadata: dict[str, str], flavor: str, filename: str = "model.pkl", pip_reqs: list[str] | str = None) -> None:
    """Register the DLHub servable with the provided metadata

    Args:
        metadata (dict): The metadata of the DLHub servable to be registered with Garden
        flavor (str): The flavor of the servable ("sklearn", "pytorch", etc.)
        filename (str): The name of the serialized model file on the DLHub server (defaults to "model.pkl")
        pip_reqs (list | str): The pip requirements for the servable, can either be a list of strings or the path to a requirements.txt file
    """
    with Loading("Instantiating GardenClient...", "Client instantiated!"):
        client = GardenClient()

    with Loading("Fetching model from DLHub...", "Model Retrieved!"):
        res = requests.get(f"{dl.base_url}servables/{metadata['dlhub']['shorthand_name']}", json={"filename": filename})

    with Loading("", "Model saved to temporary file!"):
        with open("model.pkl", "wb") as f:
            f.write(res.content)

    with Loading("Registering model with Garden...", "Model Registered!"):
        if isinstance(pip_reqs, str):
            with open(pip_reqs, "r") as f:
                pip_reqs = [line.strip() for line in f.readlines()]

        local_model = LocalModel(model_name=metadata["dlhub"]["name"],
                                 flavor=flavor,
                                 extra_pip_requirements=pip_reqs,  # or []
                                 local_path="model.pkl",
                                 user_email=client.get_email())
        registered_model = client.register_model(local_model)

    """ with Loading("Removing local model temporary file...", "Temporary file removed!"):
        os.remove("model.pkl") """

    with Loading("Defining step for pipeline creation...", "Step defined!"):
        @step
        def run_inference(input_arg: object, model=Model(registered_model.model_name)) -> object:  # object not the best type hint
            return model.predict(input_arg)

    with Loading("Creating callable pipeline...", "Pipeline Created!"):
        pipeline = client.create_pipeline(dl.get_username(),
                                          metadata["datacite"]["titles"]["title"],
                                          short_name=metadata["dlhub"]["name"],
                                          steps=(run_inference,),
                                          requirements_file=None,  # perhaps add ability to use this in addition to pip_reqs
                                          description=None,  # not a great analogy in dlhub
                                          version=metadata["dlhub"]["version"],
                                          year=datetime.utcfromtimestamp(int(metadata["dlhub"]["publication_date"])/1000).year,
                                          tags=metadata["dlhub"]["domains"])  # seems to evoke a similar idea in dlhub

    with Loading("Building container for pipeline...", "Container Built!"):
        container_uuid = client.build_container(pipeline)

    print(f"* Created container with uuid: {container_uuid}.")

    with Loading("Registering pipeline with Garden...", "Pipeline Registered!"):
        func_uuid = client.register_pipeline(pipeline, container_uuid)

    print(f"* Created function with uuid: {func_uuid}.")


def main() -> None:
    """Run the script"""
    register_model(get_dlhub_metadata("GB_2"), "sklearn", "GB_2_model.pickle", [])


if __name__ == "__main__":
    main()
