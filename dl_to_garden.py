#  Given a DLHub Servable as input, register the model it represents to Garden
#  Isaac Darling
#  May 2023

import requests
from datetime import datetime
from threading import Thread
from itertools import cycle
from time import sleep
from dlhub_sdk import DLHubClient
from garden_ai import GardenClient, Model, step

"""
The general srtucture is to:
    1. get location on disk
    2. provide to app.model.register -> use mlmodel.LocalModel and retrieve full metadata
    3. make step that calls predict on generalized model
    4. create pipeline (template stuffs)
    5. synthesize pipeline.register (client.register_pipeline)
    6. profit
    * everywhere the CLI prints something I will too
https://github.com/Garden-AI/garden/issues/112
"""

dl = DLHubClient()

class Loading:
    """Self contained class for multithreading a loading message during execution"""
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
                print(f"\r{' ' * len(self.msg)}  ", flush=True, end="")
                print(f"\r{self.complete}", flush=True)
                break
            print(f"\r{self.msg} {x}", flush=True, end="")
            sleep(0.1)


def get_dlhub_metadata(name: str) -> dict[str, str]:
    return dl.search(f"dlhub.shorthand_name: {dl.get_username()}/{name}", advanced=True, only_latest=True)[0]


def register_model(metadata: dict[str, str], flavor: str, pip_reqs: list[str] = None) -> None:
    with Loading("Fetching and registering model from DLHub...", "Model Registered!"):
        model_name = requests.get(f"{dl.base_url}/{metadata['dlhub']['shorthand_name']}", json={"flavor": flavor, "pip_reqs": pip_reqs, **metadata}).json()["full_model_name"]

    with Loading("Defining step for pipeline creation...", "Step defined!"):
        @step
        def run_inference(input_arg: object, model=Model(model_name)) -> object:  # object not the best type hint
            return model.predict(input_arg)

    with Loading("Instantiating GardenClient and creating pipeline...", "Pipeline Created!"):
        client = GardenClient()
        pipeline = client.create_pipeline(dl.get_username(),
                                          metadata["datacite"]["titles"]["title"],
                                          short_name=metadata["dlhub"]["name"],
                                          steps=(run_inference,),
                                          requirements_file=None,  # perhaps add ability to use this in addition to pip_reqs
                                          description=None,  # not a great analogy in dlhub
                                          version=metadata["dlhub"]["version"],
                                          year=datetime.utcfromtimestamp(int(metadata["dlhub"]["publication_date"])/1000).year,
                                          tags=metadata["dlhub"]["domains"]  # seems to evoke a similar idea in dlhub
                                          )

    with Loading("Building container for pipeline...", "Container Built!"):
        container_uuid = client.build_container(pipeline)

    with Loading("Registering pipeline with Garden...", "Pipeline Registered!"):
        print(f"* Created container with uuid: {container_uuid}.")
        func_uuid = client.register_pipeline(pipeline, container_uuid)

    print(f"* Created function with uuid: {func_uuid}.")

def main() -> None:
    register_model(get_dlhub_metadata("noopv10"), "sklearn")


if __name__ == "__main__":
    main()
