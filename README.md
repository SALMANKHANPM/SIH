Web Scraping Project: College Information Extraction
This project is designed to extract information about colleges from the web using Selenium and BeautifulSoup. It specifically focuses on dynamic websites like CollegeDunia.com where data is rendered after the page load, making it necessary to use Selenium for scraping.

Features
Search for College Information: Automatically searches for a specific college on DuckDuckGo and retrieves the first result from CollegeDunia.
Dynamic Content Extraction: Uses Selenium to render dynamic content and extract data from the webpage using BeautifulSoup.
Structured Data Scraping: Extracts and structures key information such as titles, meta descriptions, keywords, basic information, tabular data, and FAQs.
Saves Data in JSON: The extracted data is saved in a well-structured JSON format for easy access and analysis.
Requirements:
  Python 3.11
  Google Chrome
  Chromedriver compatible with your version of Chrome

Python packages:
  selenium
  beautifulsoup4
  requests
  duckduckgo-search
  lxml (or html.parser, depending on your preference)

Installation
  Clone the repository:

  bash
  Copy code: git clone https://github.com/yourusername/college-web-scraper.git
  cd college-web-scraper
  
  Install the required packages:
    bash
    Copy code
    pip install selenium beautifulsoup4 requests duckduckgo-search lxml

Download Chromedriver:
Download Chromedriver from here; [https://googlechromelabs.github.io/chrome-for-testing/#stable](https://googlechromelabs.github.io/chrome-for-testing/)
Running the Script

Update name variable:
Set the name variable in the script to the name of the college you want to search. For example:
python
Copy code
name = 'IIT Hyderabad'
Run the script:

Execute the Python script:
bash
Copy code
python scrape_college_info.py

Output:
The script will search for the college on DuckDuckGo, navigate to the first CollegeDunia link, extract the relevant information, and save it in a JSON file named after the college (e.g., IIT Hyderabad.json).
Extracted Data
The JSON file will contain:

Title: The title of the webpage.
Meta Description: The meta description of the page.
Keywords: Meta keywords associated with the page.
Basic Information: Structured information from the main content, including descriptions under various headers.
Tables: Tabular data extracted from the page.
FAQs: Common questions and answers related to the college.
Key Functions
extract(name, url)
Purpose: Uses Selenium to load the webpage and extract the HTML content.
Input: College name and URL of the page.
Output: Parsed HTML content as a BeautifulSoup object.
clean_text(text)
Purpose: Cleans and normalizes text by removing extra whitespace and newlines.
Input: Raw text.
Output: Cleaned text.
scrape(soup)
Purpose: Extracts structured data from the parsed HTML content.
Input: BeautifulSoup object of the webpage.
Output: A dictionary containing the extracted data.
search_college_on_duckduckgo(college_name)
Purpose: Searches for the college on DuckDuckGo and retrieves the first CollegeDunia link.
Input: College name.
Output: List of URLs found in the search result.
Notes
Headless Browsing: The script uses a headless Chrome browser for scraping. You can set options.headless = False if you want to see the browser interaction during execution.
Error Handling: Make sure to handle cases where the expected data might not be found on the page, as this can cause errors in the script.
Troubleshooting
Selenium WebDriver Errors: Ensure that the Chromedriver version matches your Chrome browser version.
Empty JSON Output: If the output JSON file is empty, check the page structure or increase the wait time (time.sleep(5)) to ensure the page has fully loaded before scraping.
Dynamic Content Not Loading: Sometimes content may not load even after the wait. In such cases, investigate the page's JavaScript and see if further interaction (e.g., scrolling or clicking) is required to load all elements.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contributing
Feel free to fork the repository and submit pull requests for improvements or additional features!
