import requests
import json
import re
import time
import os

# WARNING: To scrape 300+ bugs, you need a GitHub Personal Access Token (PAT).
# Otherwise, GitHub will rate-limit you (60 requests per hour for anonymous users).
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_TOKEN_HERE")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"} if GITHUB_TOKEN != "YOUR_TOKEN_HERE" else {}

# Expanded target list. Some repos use custom labels (e.g., 'type: bug') 
# so we provide a larger pool of repos to guarantee we hit our global target.
REPOS = [
    "facebook/react", 
    "microsoft/vscode",
    "angular/angular",
    "vuejs/core",
    "facebook/react-native",
    "sveltejs/svelte",
    "tailwindlabs/tailwindcss",
    "vercel/next.js"
]
LABELS = "bug"
MAX_PER_REPO = 100
GLOBAL_TARGET = 300 # Total reports we need across all repos

def extract_console_logs(body):
    """Simple heuristic to extract logs/stack traces from the issue description."""
    if not body: return []
    logs = []
    
    # Find code blocks (using regex `{3}` to match triple backticks safely)
    code_blocks = re.findall(r'`{3}(?:js|javascript|console|bash)?\s*(.*?)\s*`{3}', body, re.DOTALL)
    
    for block in code_blocks:
        # Look for typical error keywords
        if any(keyword in block for keyword in ["Error:", "Warning:", "Exception", "TypeError", "Failed", "Traceback"]):
            logs.append(block[:300] + "...") # Keep the first 300 characters of the log
            
    return logs

def fetch_issues(repo, needed):
    target = min(MAX_PER_REPO, needed)
    print(f"Scraping bugs from {repo} (need {target})...")
    
    issues_data = []
    page = 1
    
    while len(issues_data) < target:
        # Fetching 100 per page to reduce API calls
        url = f"https://api.github.com/repos/{repo}/issues?state=closed&labels={LABELS}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.json().get('message')}")
            break
            
        issues = response.json()
        if not issues:
            break
            
        for issue in issues:
            # Ignore Pull Requests, we only need Issues
            if 'pull_request' in issue:
                continue
                
            body = issue.get('body', '')
            logs = extract_console_logs(body)
            
            # According to the plan, discard bugs without stack traces/logs
            if not logs:
                continue
                
            issues_data.append({
                "id": f"gh_{issue['id']}",
                "title": issue['title'],
                "description": body[:500] + "..." if body and len(body) > 500 else body,
                "console_logs": logs,
                "url": issue['html_url'],
                "browser": "unknown", # Hard to reliably extract browser from GitHub Issues
                "error_type": "github_issue",
                "group": f"gh_group_{issue['id']}" # Each original bug starts its own duplicate group
            })
            
            if len(issues_data) >= target:
                break
                
        print(f"  ... page {page} processed. Found {len(issues_data)}/{target} valid reports.")
        
        # Safeguard: If a repo doesn't have enough matching issues, stop searching after 15 pages
        if page >= 15:
            print(f"  ... Reached maximum page depth (15) for {repo}. Moving to next repository.")
            break
            
        page += 1
        time.sleep(1.5) # Pause to respect rate limits
        
    return issues_data

def main():
    all_reports = []
    
    for repo in REPOS:
        needed = GLOBAL_TARGET - len(all_reports)
        if needed <= 0:
            break
            
        reports = fetch_issues(repo, needed)
        all_reports.extend(reports)
        print(f"Successfully fetched {len(reports)} reports from {repo}\n")
        
    # Save the result to the data/ directory
    os.makedirs("data", exist_ok=True)
    out_path = "data/github_issues.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
        
    print(f"Total collected: {len(all_reports)} bug reports. Saved to {out_path}")

if __name__ == "__main__":
    main()