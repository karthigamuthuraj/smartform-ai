"""
Logging utility for form filler
"""

import os
from datetime import datetime

class FormFillerLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Use single log file
        self.log_file = os.path.join(log_dir, "log.txt")
        
        self.log("=" * 80)
        self.log("Form Filler Session Started")
        self.log("=" * 80)
    
    def log(self, message, also_print=True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # Write to log.txt
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        # Print to console
        if also_print:
            print(log_message)
    
    def log_separator(self):
        """Log a separator line"""
        self.log("-" * 80)
    
    def log_section(self, title):
        """Log a section header"""
        self.log("")
        self.log("=" * 80)
        self.log(f"  {title}")
        self.log("=" * 80)
    
    def log_dict(self, data, title=None, max_items=10):
        """Log dictionary contents"""
        if title:
            self.log(f"\n{title}:")
        
        items = list(data.items())[:max_items]
        for key, value in items:
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            self.log(f"  {key} = {value_str}")
        
        if len(data) > max_items:
            self.log(f"  ... and {len(data) - max_items} more items")
    
    def log_list(self, data, title=None, max_items=10):
        """Log list contents"""
        if title:
            self.log(f"\n{title}:")
        
        items = data[:max_items]
        for i, item in enumerate(items, 1):
            self.log(f"  {i}. {item}")
        
        if len(data) > max_items:
            self.log(f"  ... and {len(data) - max_items} more items")
    
    def get_log_file(self):
        """Get the current log file path"""
        return self.log_file
