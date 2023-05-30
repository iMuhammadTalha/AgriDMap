# AgriDMap 
Made in Django app.

create virtual environment:
python -m venv venv

Then activate that environment:
venv\Scripts\activate

Then install Django in that environment:
pip install django

Then create django project named mysite:
django-admin startproject Agri



Run project using command:
python manage.py runserver

Docker: 
docker build -t agridmap .
docker run -p 8008:8008 agridmap
