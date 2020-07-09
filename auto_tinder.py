import requests
import datetime
from geopy.geocoders import Nominatim
from time import sleep
from random import random
from likeliness_classifier import Classifier
import person_detector
import tensorflow as tf
from time import time
import logging
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=f'logs/{time()}',
                    filemode='w')

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "./images/unclassified/profiles.txt"

class tinderAPI():

    def __init__(self, token):
        self._token = token

    def profile(self):
        data = requests.get(TINDER_URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def superlike(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}/super", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


class Person(object):

    def __init__(self, data, api):
        self._api = api

        self.id = data["_id"]
        self.name = data.get("name", "Unknown")

        self.bio = data.get("bio", "")
        self.distance = data.get("distance_mi", 0) / 1.60934

        self.birth_date = datetime.datetime.strptime(data["birth_date"], '%Y-%m-%dT%H:%M:%S.%fZ') if data.get(
            "birth_date", False) else None
        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]

        self.images = list(map(lambda photo: photo["url"], data.get("photos", [])))

        self.jobs = list(
            map(lambda job: {"title": job.get("title", {}).get("name"), "company": job.get("company", {}).get("name")}, data.get("jobs", [])))
        self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

        if data.get("pos", False):
            self.location = geolocator.reverse(f'{data["pos"]["lat"]}, {data["pos"]["lon"]}')


    def __repr__(self):
        return f"{self.id}  -  {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"


    def like(self):
        return self._api.like(self.id)

    def superlike(self):
        return self._api.superlike(self.id)

    def dislike(self):
        return self._api.dislike(self.id)

    def download_images(self, folder=".", sleep_max_for=0):
        with open(PROF_FILE, "r") as f:
            lines = f.readlines()
            if self.id in lines:
                return
        with open(PROF_FILE, "a") as f:
            f.write(self.id+"\r\n")
        index = -1
        for image_url in self.images:
            index += 1
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                with open(f"{folder}/{self.id}_{self.name}_{index}.jpeg", "wb") as f:
                    f.write(req.content)
            sleep(random()*sleep_max_for)

    def predict_likeliness(self, classifier, sess):
        ratings = []
        for image in self.images:
            req = requests.get(image, stream=True)
            tmp_filename = f"./images/tmp/run.jpg"
            if req.status_code == 200:
                with open(tmp_filename, "wb") as f:
                    f.write(req.content)
            img = person_detector.get_person(tmp_filename, sess)
            if img:
                img = img.convert('L')
                img.save(tmp_filename, "jpeg")
                certainty = classifier.classify(tmp_filename)
                pos = certainty["positive"]
                ratings.append(pos)
        ratings.sort(reverse=True)
        ratings = ratings[:5]
        if len(ratings) == 0:
            return 0.001
        return ratings[0]*0.6 + sum(ratings[1:])/len(ratings[1:])*0.4



class Profile(Person):

    def __init__(self, data, api):

        super().__init__(data["user"], api)

        self.email = data["account"].get("email")
        self.phone_number = data["account"].get("account_phone_number")

        self.age_min = data["user"]["age_filter_min"]
        self.age_max = data["user"]["age_filter_max"]

        self.max_distance = data["user"]["distance_filter"]
        self.gender_filter = ["Male", "Female"][data["user"]["gender_filter"]]


if __name__ == "__main__":
    token = "a4f03f05-6795-4b08-9e20-b18cb0acbcf3"
    api = tinderAPI(token)

    detection_graph = person_detector.open_graph()
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:

            classifier = Classifier(graph="./tf/training_output/retrained_graph.pb",
                                    model_path="./train-results/MobileNetV2-person-classifier.h5",
                                    labels="./tf/training_output/retrained_labels.txt")

            end_time = time() + 60*60*2.8
            while time() < end_time:
                try:
                    logging.info(f"------ TIME LEFT: {(end_time - time())/60} min -----")

                    persons = api.nearby_persons()
                    pos_schools = ["Universität Zürich", "University of Zurich", "UZH", "HWZ Hochschule für Wirtschaft Zürich",
                                   "ETH Zürich", "ETH Zurich", "ETH", "ETHZ", "Hochschule Luzern", "HSLU", "ZHAW",
                                   "Zürcher Hochschule für Angewandte Wissenschaften", "Universität Bern", "Uni Bern",
                                   "PHLU", "PH Luzern", "Fachhochschule Luzern", "Eidgenössische Technische Hochschule Zürich"]

                    for person in persons:
                        score = person.predict_likeliness(classifier, sess)

                        for school in pos_schools:
                            if school in person.schools:
                                logging.info("")
                                score *= 1.2

                        logging.info("-------------------------")
                        logging.info(f"ID: {person.id}")
                        logging.info(f"Name: {person.name}")
                        logging.info(f"BIO: {person.bio}")
                        logging.info(f"Schools: {person.schools}")
                        logging.info(f"Images: {person.images}")
                        logging.info(score)

                        if score > 0.81:
                            res = person.superlike()
                            logging.info("SUPER LIKE")
                            logging.info(f"Response: {res}")
                        elif score > 0.69:
                            res = person.like()
                            logging.info("LIKE")
                            logging.info(f"Response: {res}")
                        else:
                            res = person.dislike()
                            logging.info("DISLIKE")
                            logging.info(f"Response: {res}")
                except Exception:
                    pass




    classifier.close()
