# dermatologic_classification
Skin type/ Acne classification



### hugging face

- repo (lfs)  https://huggingface.co/afscomercial/dermatologic
- project folder: models

### Clone the Repository

1. Navigate to the desired parent directory:
```sh
    cd /path/to/parent/directory
```

2. Clone the repository using Git:
```sh
    git clone https://huggingface.co/afscomercial/dermatologic
```

3. Navigate into the cloned repository:
```sh
    cd dermatologic
```

### hugging face commands
```sh
    huggingface-cli version
```
```sh
   huggingface-cli whoami
```
```sh
    huggingface-cli logout
```

> **Note:** Ensure you have the `huggingface-cli` installed and you are logged in to your Hugging Face account before running these commands. Parent folder is a git repository and the Models subdirectory is a huggingface lfs repository.


### Setup Virtual Environment and Install Dependencies

1. Create a virtual environment:
```sh
    python -m venv env
```

2. Activate the virtual environment:
    - On Windows:
```sh
    .\env\Scripts\activate
```
    - On macOS and Linux:
```sh
    source env/bin/activate
```

3. Install `huggingface_hub`:
```sh
    pip install huggingface_hub
```

4. Verify the installation:
 ```sh
    huggingface-cli version
```

> **Note:** Ensure you have Python installed on your system before creating a virtual environment.