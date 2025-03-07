from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

# Set up Selenium
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Base URL for Mayo Clinic
INDEX_URL = "https://www.mayoclinic.org/diseases-conditions/index"

# Step 1: Get all disease links from the index pages
def get_disease_links():
    disease_links = []

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        url = f"{INDEX_URL}?letter={letter}"
        print(f"\n🔎 Fetching: {url}")

        driver.get(url)
        time.sleep(5)  # Allow JavaScript to load

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract disease links using the correct selector
        for link in soup.select("div.cmp-result-name a"):
            href = link.get("href")
            if href and "diseases-conditions" in href and href not in disease_links:
                disease_links.append(href)

        time.sleep(2)  # Avoid overloading the server

    print(f"\n✅ Found {len(disease_links)} disease pages to scrape.")
    return disease_links

# Step 2: Scrape data from each disease page
def scrape_disease_data(url):
    print(f"\n🩺 Scraping: {url}")

    try:
        driver.set_page_load_timeout(15)  # Set timeout to 15 seconds
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load
    except Exception as e:
        print(f"⚠️ Timeout Error: Skipping {url}")
        return None  # Skip this page if it times out

    soup = BeautifulSoup(driver.page_source, "html.parser")

    try:
        title = soup.find("h1").text.strip()
    except:
        title = "Unknown"

    symptoms = extract_section(soup, "Symptoms")
    causes = extract_section(soup, "Causes")
    treatment = extract_section(soup, "Treatment")

    # Remove "Request an appointment" from symptoms
    if "Request an appointment" in symptoms:
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
    disease_links = get_disease_links()

    if not disease_links:
        print("\n❌ No diseases found. Exiting.")
        return

    all_diseases = []

    for idx, link in enumerate(disease_links):  # Process ALL links
        data = scrape_disease_data(link)
        if data:
            all_diseases.append(data)
        time.sleep(2)

    # Save to JSON file
    with open("../data/mayo_disease_data.json", "w") as f:
        json.dump(all_diseases, f, indent=4)

    print("\n🎉 Scraping complete! Data saved to mayo_disease_data.json.")
    driver.quit()

if __name__ == "__main__":
    main()
