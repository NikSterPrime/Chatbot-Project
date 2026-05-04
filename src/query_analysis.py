"""
Query Analysis and Visualization Tool
Shows the relationship between queries and responses, including fallback tracking.
"""

import json
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Sample data structure for analysis
class QueryAnalyzer:
    def __init__(self):
        self.query_responses = []
        self.fallback_count = 0
        self.success_count = 0
        self.intent_counts = defaultdict(int)
        self.fallback_by_intent = defaultdict(int)
        
    def add_query_response(self, query, detected_intent, response, is_fallback=False):
        """
        Track a query and its response
        
        Args:
            query: User's input query
            detected_intent: Intent detected by the model
            response: Response given by chatbot
            is_fallback: Whether this was a fallback response
        """
        entry = {
            'timestamp': datetime.now(),
            'query': query,
            'detected_intent': detected_intent,
            'response': response,
            'is_fallback': is_fallback
        }
        self.query_responses.append(entry)
        
        if is_fallback:
            self.fallback_count += 1
        else:
            self.success_count += 1
            
        self.intent_counts[detected_intent] += 1
        if is_fallback:
            self.fallback_by_intent[detected_intent] += 1
    
    def get_fallback_rate_by_intent(self):
        """Calculate fallback rate for each intent"""
        rates = {}
        for intent, count in self.intent_counts.items():
            fallback = self.fallback_by_intent.get(intent, 0)
            rate = (fallback / count * 100) if count > 0 else 0
            rates[intent] = {
                'total_queries': count,
                'fallback_count': fallback,
                'fallback_rate': rate
            }
        return rates
    
    def load_sample_data(self):
        """Load sample query-response data for demonstration"""
        print("Loading sample data...")
        
        # Sample queries with their intents
        sample_data = [
            ("hi", "greeting", "Hi there. Great to have you here.", False),
            ("hello", "greeting", "Hello. How can I help today?", False),
            ("goodbye", "goodbye", "Goodbye. Take care.", False),
            ("what's the weather", "fallback", "I'm not sure about that.", True),
            ("thank you", "gratitude", "You're welcome!", False),
            ("thanks", "gratitude", "Happy to help!", False),
            ("how are you", "greeting", "I'm doing great!", False),
            ("see you", "goodbye", "See you soon.", False),
            ("can you code", "fallback", "I'm not sure I can help with that.", True),
            ("bye bye", "goodbye", "Take care and have a great day.", False),
            ("tell me a joke", "fallback", "I'm not sure how to help with that.", True),
            ("help", "help", "I can assist with greetings, goodbyes, thanks, and more.", False),
            ("what is your name", "fallback", "I don't have personal information.", True),
            ("hello there", "greeting", "Welcome. What would you like to do?", False),
            ("thanks a lot", "gratitude", "You're welcome!", False),
            ("i'm leaving", "goodbye", "Take care and have a great day.", False),
            ("who are you", "fallback", "I'm a chatbot here to help.", True),
            ("hey", "greeting", "Hey. Ready when you are.", False),
            ("thanks buddy", "gratitude", "Happy to help!", False),
            ("what time is it", "fallback", "I don't have access to time.", True),
            ("appreciate it", "gratitude", "You're welcome!", False),
            ("morning", "greeting", "Good morning! How can I help?", False),
            ("how do i learn programming", "fallback", "That's beyond my current capabilities.", True),
            ("thank you so much", "gratitude", "You're welcome!", False),
            ("goodbye friend", "goodbye", "See you soon.", False),
        ]
        
        for query, intent, response, is_fallback in sample_data:
            self.add_query_response(query, intent, response, is_fallback)
    
    def visualize(self):
        """Create comprehensive visualizations of query-response data"""
        
        if not self.query_responses:
            print("No data to visualize!")
            return
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Chatbot Query Analysis & Fallback Tracking', fontsize=16, fontweight='bold')
        
        # 1. Overall Fallback vs Success Pie Chart
        ax1 = axes[0, 0]
        labels = ['Successful Intent Match', 'Fallback Responses']
        sizes = [self.success_count, self.fallback_count]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.1)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        ax1.set_title('Overall Response Distribution', fontweight='bold')
        
        # 2. Queries by Intent Bar Chart
        ax2 = axes[0, 1]
        intents = list(self.intent_counts.keys())
        counts = [self.intent_counts[i] for i in intents]
        colors_bar = ['#3498db' if i != 'fallback' else '#e74c3c' for i in intents]
        
        bars = ax2.barh(intents, counts, color=colors_bar)
        ax2.set_xlabel('Number of Queries', fontweight='bold')
        ax2.set_title('Queries by Intent Type', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # 3. Fallback Rate by Intent
        ax3 = axes[1, 0]
        rates = self.get_fallback_rate_by_intent()
        intents_sorted = sorted(rates.keys(), key=lambda x: rates[x]['fallback_rate'], reverse=True)
        fallback_rates = [rates[i]['fallback_rate'] for i in intents_sorted]
        colors_rate = ['#e74c3c' if rate > 50 else '#f39c12' if rate > 0 else '#2ecc71' 
                      for rate in fallback_rates]
        
        bars2 = ax3.barh(intents_sorted, fallback_rates, color=colors_rate)
        ax3.set_xlabel('Fallback Rate (%)', fontweight='bold')
        ax3.set_title('Fallback Rate by Intent', fontweight='bold')
        ax3.set_xlim(0, 100)
        ax3.grid(axis='x', alpha=0.3)
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax3.text(width + 2, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold')
        
        # 4. Fallback vs Success per Intent
        ax4 = axes[1, 1]
        df_data = []
        for intent in sorted(self.intent_counts.keys()):
            total = self.intent_counts[intent]
            fallback = self.fallback_by_intent.get(intent, 0)
            success = total - fallback
            df_data.append({'Intent': intent, 'Successful': success, 'Fallback': fallback})
        
        df = pd.DataFrame(df_data)
        df.set_index('Intent')[['Successful', 'Fallback']].plot(
            kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'], width=0.7)
        ax4.set_title('Success vs Fallback per Intent', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_xlabel('Intent', fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print a detailed text summary of the analysis"""
        print("\n" + "="*60)
        print("QUERY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Queries: {len(self.query_responses)}")
        print(f"Successful Intent Matches: {self.success_count}")
        print(f"Fallback Responses: {self.fallback_count}")
        print(f"Overall Success Rate: {(self.success_count/len(self.query_responses)*100):.1f}%")
        print(f"Overall Fallback Rate: {(self.fallback_count/len(self.query_responses)*100):.1f}%")
        print("\n" + "-"*60)
        print("BREAKDOWN BY INTENT:")
        print("-"*60)
        
        rates = self.get_fallback_rate_by_intent()
        for intent in sorted(rates.keys()):
            data = rates[intent]
            print(f"\n{intent.upper()}:")
            print(f"  Total Queries: {data['total_queries']}")
            print(f"  Fallback Count: {data['fallback_count']}")
            print(f"  Fallback Rate: {data['fallback_rate']:.1f}%")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main function to run the query analyzer"""
    print("\n" + "="*60)
    print("CHATBOT QUERY ANALYSIS TOOL")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = QueryAnalyzer()
    
    # Load sample data
    analyzer.load_sample_data()
    
    # Print summary
    analyzer.print_summary()
    
    # Visualize the data
    print("Generating visualizations...")
    analyzer.visualize()


if __name__ == "__main__":
    main()
