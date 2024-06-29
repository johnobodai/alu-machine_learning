import requests

def availableShips(passengerCount):
    url = 'https://swapi.dev/api/starships/'
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data from {url}: {response.status_code}")
            return []

        data = response.json()
        results = data.get('results', [])

        for ship in results:
            if ship['passengers'].isdigit() and int(ship['passengers']) >= passengerCount:
                ships.append(ship['name'])

        url = data['next']

    return ships
