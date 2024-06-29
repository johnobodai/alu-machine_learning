#!/usr/bin/env python3
"""Retrieves home planets of sentient species."""
import requests


def sentientPlanets():
    """Returns list of home planet names."""
    url = "https://swapi.dev/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data['results']:
            if species['classification'] == 'sentient' or species['designation'] == 'sentient':
                homeworld = species['homeworld']
                if homeworld:
                    planet_response = requests.get(homeworld)
                    planet_data = planet_response.json()
                    planets.append(planet_data['name'])

        url = data['next']

    return planets
