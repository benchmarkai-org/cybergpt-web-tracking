import numpy as np
from urllib.parse import urlparse


def jensen_shannon_distance(p: np.array, q: np.array, epsilon: float = 1e-10) -> float:
    """Compute the Jensen-Shannon distance between two probability distributions."""
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    m = 0.5 * (p_norm + q_norm)

    js_divergence = 0.5 * np.sum(
        p_norm * np.log(p_norm / (m + epsilon) + epsilon)
    ) + 0.5 * np.sum(q_norm * np.log(q_norm / (m + epsilon) + epsilon))
    js_distance = np.sqrt(max(0, js_divergence))
    return float(js_distance)


def normalise_domain(url: str) -> str:
    """Extract and normalise domain from URL."""
    try:
        if not url.startswith("http"):
            url = f"http://{url}"

        # Parse URL and get netloc (domain part)
        domain = urlparse(url).netloc

        # Remove www. if present
        if domain.startswith("www."):
            domain = domain[4:]

        # Special handling for googlevideo.com domains
        if "googlevideo.com" in domain:
            return "googlevideo.com"

        # Split domain and get main domain + TLD
        parts = domain.split(".")

        # Handle subdomains
        if len(parts) > 2:
            # Special cases for known services where we want to preserve the subdomain
            special_cases = ["googleusercontent"]
            if any(case in domain for case in special_cases):
                return domain

            # For other cases, take only the main domain and TLD
            return ".".join(parts[-2:])

        return domain
    except:
        return url
