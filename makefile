v-env:
	python3 -m venv venv
	. ./venv/bin/activate
	. ./venv/bin/activate; python -m pip install -r requirements.txt
	deactivate

	echo "Done creating virtual environment and instaling dependencies. To run, please use the following command:\n . ./venv/bin/activate; python3 main.py; deactivate"
