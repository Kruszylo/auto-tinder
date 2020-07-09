import requests
import datetime
from geopy.geocoders import Nominatim
from time import sleep
from random import random
from likeliness_classifier import Classifier
import person_detector
import tensorflow as tf
from time import time
from auto_tinder import tinderAPI

TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "./images/unclassified/profiles.txt"


if __name__ == "__main__":
    token = "90e1a8f5-6602-43fe-a151-a96acde451b6"
    api = tinderAPI(token)

    while True:
        persons = api.nearby_persons()
        print(f'persons collected: {len(persons)}')
        for person in persons:
            person.download_images(folder="./images/unclassified", sleep_max_for=random()*3)
            sleep(random()*10)
        sleep(random()*10)