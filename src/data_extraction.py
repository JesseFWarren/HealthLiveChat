from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

# Path to JSON file to save progress
OUTPUT_FILE = "../data/mayo_disease_data.json"

# Set up Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Base URL for Mayo Clinic
INDEX_URL = "https://www.mayoclinic.org/diseases-conditions/index"

# Load existing progress
def load_existing_data():
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Return empty dict if file doesn't exist

# Save progress after each letter
def save_progress(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Step 1: Get all disease links from the index pages
def get_disease_links():
    existing_data = load_existing_data()
    disease_links_by_letter = {}

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in existing_data:
            print(f"Skipping {letter}, already scraped.")
            continue  # Skip already completed letters

        url = f"{INDEX_URL}?letter={letter}"
        print(f"\nFetching: {url}")

        driver.get(url)
        time.sleep(5)  # Allow JavaScript to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        disease_links = []

        # Extract disease links
        for link in soup.select("div.cmp-result-name a"):
            href = link.get("href")
            if href and "diseases-conditions" in href:
                disease_links.append(href)

        if disease_links:
            disease_links_by_letter[letter] = disease_links

        time.sleep(2)  # Avoid overloading the server

    return disease_links_by_letter

# Step 2: Scrape data from each disease page with retries
def scrape_disease_data(url, max_retries=3):
    print(f"\nScraping: {url}")

    for attempt in range(max_retries):
        try:
            driver.set_page_load_timeout(15)  # Set timeout to 15 seconds
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to load

            soup = BeautifulSoup(driver.page_source, "html.parser")

            title = soup.find("h1").text.strip() if soup.find("h1") else "Unknown"
            symptoms = extract_section(soup, "Symptoms")
            causes = extract_section(soup, "Causes")
            treatment = extract_section(soup, "Treatment")

            # Remove unwanted text
            symptoms = symptoms.replace("Request an appointment", "").strip()
            symptoms = symptoms if symptoms else "Not Available"
            causes = causes if causes else "Not Available"
            treatment = treatment if treatment else "Not Available"

            return {
                "disease": title,
                "symptoms": symptoms,
                "causes": causes,
                "treatment": treatment
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(5)  # Wait before retrying

    print(f"Failed to scrape {url} after {max_retries} attempts.")
    return None

# Helper function to extract sections
def extract_section(soup, section_name):
    """Extracts text content under a specific section heading."""
    section = soup.find(["h2", "h3"], string=lambda text: text and section_name in text)
    
    if section:
        content = section.find_next("div")
        if content:
            text = " ".join(content.stripped_strings)

            # Remove unwanted text
            ignore_phrases = ["Request an appointment", "Book an appointment"]
            for phrase in ignore_phrases:
                text = text.replace(phrase, "")

            return text.strip()

    return "Not Available"

# Step 3: Run the scraper and save results
def main():
    existing_data = load_existing_data()
    disease_links_by_letter = get_disease_links()

    if not disease_links_by_letter:
        print("\nNo diseases found. Exiting.")
        return

    for letter, links in disease_links_by_letter.items():
        if letter in existing_data:
            print(f"Skipping {letter}, already scraped.")
            continue  # Skip already scraped letters

        print(f"\nScraping diseases starting with {letter}...")
        existing_data[letter] = []

        for link in links:
            data = scrape_disease_data(link)
            if data:
                existing_data[letter].append(data)
            time.sleep(2)

        # Save after each letter
        save_progress(existing_data)
        print(f"Saved progress for {letter}.")

    print("\nScraping complete! Data saved to mayo_disease_data.json.")
    driver.quit()

if __name__ == "__main__":
    main()
