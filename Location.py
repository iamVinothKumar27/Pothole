import requests
import webbrowser

def get_current_location():
    try:
        # Send a request to the IP info API
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()

            # Extracting required fields
            ip = data.get('ip')
            city = data.get('city')
            region = data.get('region')
            country = data.get('country')
            loc = data.get('loc')  # Format: "latitude,longitude"
            org = data.get('org')
            timezone = data.get('timezone')

            # Print location details
            print("Your Location Details:")
            print(f"IP Address : {ip}")
            print(f"City       : {city}")
            print(f"Region     : {region}")
            print(f"Country    : {country}")
            print(f"Latitude/Longitude: {loc}")
            print(f"Organization: {org}")
            print(f"Timezone   : {timezone}")

            # Open location in Google Maps
            if loc:
                gmaps_url = f"https://www.google.com/maps?q={loc}"
                webbrowser.open(gmaps_url)
                print(f"Opening your location in Google Maps: {gmaps_url}")
            else:
                print("Location coordinates not found.")
        else:
            print("Failed to retrieve location data. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)

# Run the function
get_current_location()
