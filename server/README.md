### Creating a Python Virtual Environment

To create a Python virtual environment, follow these steps:

1. **Navigate to your project directory**:
    ```sh
    cd /path/to/your/project
    ```

2. **Create the virtual environment**:
    ```sh
    python -m venv env
    ```
    Replace `env` with your preferred name for the virtual environment.

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source env/bin/activate
        ```

4. **Install necessary packages**:
    ```sh
    pip install -r requirements.txt
    ```
    Ensure you have a `requirements.txt` file with the list of dependencies.

5. **Deactivate the virtual environment**:
    ```sh
    deactivate
    ```

6. **Freeze requirements**:
    ```sh
    pip freeze > requirements.txt
    ```
    This command will save the current state of your installed packages to the `requirements.txt` file.

Now you have a virtual environment set up for your project.

### Running the Flask Application

To run the Flask application, follow these steps:

1. **Set the Flask application environment variable**:
    ```sh
    export FLASK_APP=app.py
    ```
    Replace `app.py` with the name of your main Flask application file.

2. **Run the Flask application**:
    ```sh
    flask run
    ```

3. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000/` to see your Flask application in action.

Now your Flask application should be running and accessible in your web browser.

### Running the Flask Application with Gunicorn

To run the Flask application with Gunicorn, follow these steps:

1. **Install Gunicorn**:
    ```sh
    pip install gunicorn
    ```

2. **Run the Flask application with Gunicorn**:
    ```sh
    gunicorn --timeout 120 -w 4 app:app
    ```
    Replace `app:app` with the module and application name of your Flask app. The `-w 4` flag specifies the number of worker processes.
    The `-b` flag specifies the address and port on which the Gunicorn server should listen for incoming requests.

Now your Flask application should be running with Gunicorn and accessible in your web browser.

### Render

1. **Configure the service**:
    - Set the build and start commands:
        - **Build Command**: 
            ```sh
            pip install -r requirements.txt
            ```
        - **Start Command**: 
            ```sh
            gunicorn --timeout 120 -w 4 app:app
            ```
            Replace `app:app` with the module and application name of your Flask app.


url: https://dermatologic-classification.onrender.com

### pyenv
To manage multiple Python versions using `pyenv`, follow these steps:

1. **Install `pyenv`**:
    Follow the installation instructions from the [pyenv GitHub repository](https://github.com/pyenv/pyenv#installation).

2. **Install a specific Python version**:
    ```sh
    pyenv install 3.11.5
    ```
    Replace `3.11.5` with the version of Python you need.

3. **Set the local Python version for your project**:
    ```sh
    pyenv local 3.11.5
    ```
    This command will create a `.python-version` file in your project directory with the specified Python version.

4. **Verify the Python version**:
    ```sh
    python --version
    ```
    Ensure that the output matches the version you set with `pyenv`.
```
