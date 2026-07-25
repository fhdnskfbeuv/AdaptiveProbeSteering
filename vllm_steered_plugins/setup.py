from setuptools import setup, find_packages

setup(
    name="vllm-steered-plugins",
    version="0.1.0",
    description="vLLM v1 plugin registering the probe-steered Llama model.",
    packages=find_packages(),
    install_requires=["vllm"],
    entry_points={
        "vllm.general_plugins": [
            "steered_registry = vllm_steered_plugins:register_models",
        ],
    },
    python_requires=">=3.8",
)
