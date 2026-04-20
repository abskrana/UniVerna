import os
import time
import copy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup, NavigableString

# Define the sections to be extracted
SECTIONS_TO_EXTRACT =[
    "Details", 
    "Benefits", 
    "Eligibility", 
    "Application Process", 
    "Documents Required"
]

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_clean_text(element):
    elem_copy = copy.copy(element)
    for br in elem_copy.find_all("br"):
        br.replace_with("\n")
    for tag in elem_copy.find_all(['p', 'div', 'li', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4']):
        tag.insert_before('\n')
        tag.insert_after('\n')
    text = elem_copy.get_text()
    lines =[line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def get_scheme_links(driver, main_url, max_links=10):
    print(f"Loading main page: {main_url}")
    driver.get(main_url)
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except Exception:
        print("Table did not load in time.")
        return[]
    time.sleep(2)
    link_elements = driver.find_elements(By.XPATH, "//table/tbody/tr/td[last()]//a")
    urls =[]
    for elem in link_elements:
        href = elem.get_attribute('href')
        if href and href not in urls:
            urls.append(href)
        if len(urls) >= max_links:
            break
    return urls

def scrape_scheme_data(driver, url):
    driver.get(url)
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
    except Exception:
        print(f"Timeout waiting for page content at {url}")
        return None

    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    extracted_data = {}
    
    # 1. Extract Scheme Name (Title)
    title = "Unknown Scheme Name"
    title_tag = None
    for h1 in soup.find_all('h1'):
        text = h1.get_text(strip=True)
        if text and text.lower() != "myscheme":
            title_tag = h1
            break
    if not title_tag:
        h2_tags = soup.find_all('h2')
        if h2_tags:
            title_tag = max(h2_tags, key=lambda t: len(t.get_text(strip=True)))
    if title_tag:
        title = title_tag.get_text(strip=True)
    extracted_data['Title'] = title
    
    # 2. Extract State/Ministry
    state = "Unknown State/Ministry"
    if title_tag:
        prev_sibling = title_tag.find_previous_sibling()
        if prev_sibling and len(prev_sibling.get_text(strip=True)) > 0:
            state = prev_sibling.get_text(strip=True)
        else:
            parent = title_tag.parent
            if parent:
                parent_prev = parent.find_previous_sibling()
                if parent_prev and len(parent_prev.get_text(strip=True)) > 0:
                    state = parent_prev.get_text(strip=True)
    if state in ['|', '-']: 
        state = "Unknown State/Ministry"
    extracted_data['State'] = state

    # 3. Extract Tags (between Title and Details)
    tags =[]
    if title_tag:
        title_strings = list(title_tag.stripped_strings)
        curr = title_tag.next_element
        while curr:
            # Stop if we reach a main heading like "Details"
            if getattr(curr, 'name', None) in ['h2', 'h3']:
                if 'details' in curr.get_text(strip=True).lower() or 'benefits' in curr.get_text(strip=True).lower():
                    break
                    
            if isinstance(curr, NavigableString):
                text = curr.strip()
                if text and 0 < len(text) < 40:
                    ignore_list =['check eligibility', 'share', 'print', 'login', 'sign in', 'apply', 'details', 'bookmark']
                    if text.lower() not in ignore_list and text not in tags and text not in title_strings:
                        tags.append(text)
            
            # Failsafe to prevent endless tag collection
            if len(tags) > 15:
                break
            curr = curr.next_element
    extracted_data['Tags'] = tags

    # 4. Extract the Sections
    known_sections =[s.lower() for s in SECTIONS_TO_EXTRACT] + ["frequently asked questions", "sources and references"]
    for section in SECTIONS_TO_EXTRACT:
        header = None
        for tag in soup.find_all(['h2', 'h3', 'h4', 'div']):
            if tag.get_text(strip=True).lower() == section.lower():
                if not tag.find_parent(['nav', 'ul', 'li']) and tag.name != 'a':
                    header = tag
                    break
        
        section_text = ""
        if header:
            content_pieces =[]
            current = header
            while current:
                siblings = current.find_next_siblings()
                if siblings:
                    is_stop_element = False
                    for sibling in siblings:
                        heading_tags = sibling.find_all(['h2', 'h3'])
                        if sibling.name in['h2', 'h3']:
                            heading_tags.insert(0, sibling)
                        for h_tag in heading_tags:
                            h_text = h_tag.get_text(strip=True).lower()
                            if h_text in known_sections and h_text != section.lower():
                                is_stop_element = True
                                break
                        if is_stop_element:
                            break 
                        text = get_clean_text(sibling)
                        if text:
                            content_pieces.append(text)
                    if is_stop_element:
                        break 
                    break 
                else:
                    current = current.parent
                    if current and current.name in ['body', 'html', 'main']:
                        break
            section_text = "\n".join(content_pieces)
        extracted_data[section] = section_text if section_text else "Information not available."
    return extracted_data

def save_to_txt(data, index, output_dir="Scheme_Data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    safe_title = "".join([c for c in data['Title'][:30] if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filename = f"{output_dir}/Scheme_{index:02d}_{safe_title.replace(' ', '_')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{data['State']}\n")
        f.write(f"{data['Title']}\n\n")
        
        # Write tags dynamically right between title and details
        if data.get('Tags'):
            f.write("Tags\n")
            f.write(f"{', '.join(data['Tags'])}\n\n")
            
        for section in SECTIONS_TO_EXTRACT:
            f.write(f"{section}\n")
            f.write(f"{data[section]}\n\n")
    print(f"Saved: {filename}")

def main():
    main_url = 'https://rules.myscheme.in/'
    driver = setup_driver()
    try:
        print("Fetching scheme links...")
        scheme_urls = get_scheme_links(driver, main_url, max_links=100)
        if not scheme_urls:
            print("No links found.")
            return
        print(f"Found {len(scheme_urls)} links. Starting extraction...")
        for index, url in enumerate(scheme_urls, start=1):
            print(f"[{index}/100] Extracting data from {url} ...")
            data = scrape_scheme_data(driver, url)
            if data:
                save_to_txt(data, index)
            else:
                print(f"Failed to extract data for scheme {index}.")
    finally:
        driver.quit()
        print("Scraping completed. Browser closed.")

if __name__ == "__main__":
    main()