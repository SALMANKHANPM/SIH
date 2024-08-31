from duckduckgo_search import DDGS

def search_college_on_duckduckgo(college_name):
    query = f"{college_name} site:collegedunia.com"
    
    # Create a DDGS object and perform a search
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=1)
        
    if results:
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Description: {result['body']}")
            return result['href']
    else:
        return "No CollegeDunia link found in the search results."

if __name__ == "__main__":
    colleges = ["IIT Madras", "JNTU", "JNU University"]
    for college in colleges:
        print(f"Searching for: {college}")
        first_link = search_college_on_duckduckgo(college)
        print(f"First CollegeDunia link: {first_link}\n")
