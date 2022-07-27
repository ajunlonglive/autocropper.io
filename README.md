# photoscan_splitter

This is a very simple Flask app that let's the user upload a image and detects how many (if any) faces are there in the picture.

## Hosting on Heroku

Try it [here](https://scanned-photo-split.herokuapp.com/).

##### Generate pipfile with command:

`pipenv install opencv-contrib-python`

`pipenv install Flask`

`pipenv install gunicorn`

`pipenv install numpy`

##### Add the following buildpack:

https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt

and include a list of apt package names to be installed in the `Aptfile` (more details here: https://stackoverflow.com/a/51004957/660711)