import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging
from pandasai import Agent
from pandasai.llm import OpenAI
import warnings
import numpy as np
import random
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateSetMerger:
    def __init__(self):
        """Initialize database connection and PandasAI setup"""
        load_dotenv()
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('POSTGRES_DB', 'ratemodel')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', '123')

        self.connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("✅ Database engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            self.engine = None

        # Setup PandasAI
        self.llm = self._setup_openai_llm()
        self.agent = None
        self.current_data = None
        
        # Flag to use mock data if database connection fails
        self.use_mock_data = False

    def _setup_openai_llm(self):
        """Setup OpenAI LLM for PandasAI"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found - AI features will be disabled")
                return None

            llm = OpenAI(
                api_token=api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            logger.info("✅ OpenAI LLM configured successfully!")
            return llm
        except Exception as e:
            logger.error(f"OpenAI LLM setup failed: {e}")
            return None

    def _generate_mock_data(self):
        """Generate mock data for testing when database is not available"""
        logger.info("Generating mock data for testing...")
        
        # Practice areas
        practice_areas = ['Litigation', 'Corporate', 'Tax', 'Real Estate', 'IP', 'Employment']
        
        # Rate classes
        rate_classes = ['Partner', 'Associate', 'Senior Associate', 'Of Counsel', 'Paralegal']
        
        # Generate 20 mock rate sets
        mock_data = []
        
        for i in range(1, 21):
            # Generate base rate amount (between $300 and $800)
            base_rate = random.uniform(300, 800)
            
            # Select 1-2 practice areas
            num_areas = random.randint(1, 2)
            selected_areas = '|'.join(random.sample(practice_areas, num_areas))
            
            # Select 1-3 rate classes
            num_classes = random.randint(1, 3)
            selected_classes = '|'.join(random.sample(rate_classes, num_classes))
            
            # Create similar rate sets with small variations
            if i % 2 == 0:  # Create pairs of similar rate sets
                # Adjust rate by small percentage (within 5%)
                rate_adjustment = base_rate * random.uniform(-0.04, 0.04)
                adjusted_rate = base_rate + rate_adjustment
                
                # Use same practice areas as previous
                practice_area = mock_data[i-2]['practice_areas']
                
                # Similar client counts
                client_count = mock_data[i-2]['client_count'] + random.randint(-1, 1)
                client_count = max(1, client_count)  # Ensure at least 1 client
                
                # Similar matter counts
                matter_count = mock_data[i-2]['matter_count'] + random.randint(-2, 2)
                matter_count = max(0, matter_count)  # Ensure non-negative
            else:
                adjusted_rate = base_rate
                practice_area = selected_areas
                client_count = random.randint(1, 10)
                matter_count = random.randint(0, 20)
            
            mock_data.append({
                'rate_set_id': i,
                'rate_set_name': f"Rate Set {i}",
                'rate_set_code': f"RS{i:03d}",
                'final_rate_amount': adjusted_rate,
                'discount_structure': selected_classes,
                'practice_areas': practice_area,
                'assigned_clients': f"Client {i}|Client {i+1}" if client_count > 1 else f"Client {i}",
                'matter_count': matter_count,
                'client_count': client_count
            })
        
        logger.info(f"Generated {len(mock_data)} mock rate sets")
        return pd.DataFrame(mock_data)

    def capture_key_attributes(self):
        """
        Step 1: For each existing rate set, capture key attributes
        (rate amounts, discount structure, matter or client exception, etc.)
        """
        if not self.engine or self.use_mock_data:
            logger.warning("Database connection not available or using mock data - generating sample data instead")
            return self._generate_mock_data()
            
        query = """
        SELECT
            rs.rate_set_id,
            rs.rate_set_name,
            rs.rate_set_code,
            AVG(rd.rate_amount) as final_rate_amount,
            STRING_AGG(DISTINCT rd.rate_class, '|') as discount_structure,
            STRING_AGG(DISTINCT rd.practice_group, '|') as practice_areas,
            STRING_AGG(DISTINCT c.client_name, '|') as assigned_clients,
            COUNT(DISTINCT m.matter_id) as matter_count,
            COUNT(DISTINCT c.client_id) as client_count
        FROM rate_set rs
        LEFT JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        LEFT JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        LEFT JOIN client c ON rs.rate_set_id = c.rate_set_id AND c.is_active = 'Y'
        LEFT JOIN matter m ON rs.rate_set_id = m.rate_set_id AND m.is_active = 'Y'
        WHERE rd.rate_amount IS NOT NULL AND rd.rate_amount > 0
        GROUP BY rs.rate_set_id, rs.rate_set_name, rs.rate_set_code
        HAVING AVG(rd.rate_amount) IS NOT NULL
        """

        logger.info("Capturing key attributes for all rate sets...")
        try:
            # Use the engine directly with a string query
            rate_sets = pd.read_sql_query(query, self.engine)
            if rate_sets.empty:
                logger.warning("No rate sets found in database - switching to mock data")
                return self._generate_mock_data()
                
            logger.info(f"Captured attributes for {len(rate_sets)} rate sets")
            return rate_sets
        except Exception as e:
            logger.error(f"Error capturing rate set attributes: {e}")
            logger.warning("Switching to mock data for demonstration")
            self.use_mock_data = True
            return self._generate_mock_data()

    def compare_sets_and_identify_candidates(self, rate_sets, threshold_percent=5.0):
        """
        Step 2: Compare sets: If sets differ by <5% in final rate amounts
        but are assigned to similar matter/client contexts, mark them as "merge candidates."
        """
        logger.info(f"Comparing rate sets with {threshold_percent}% threshold...")
        
        # Check if rate_sets is empty
        if rate_sets.empty:
            logger.warning("No rate sets available for comparison")
            return pd.DataFrame()
            
        merge_candidates = []

        for i, set1 in rate_sets.iterrows():
            for j, set2 in rate_sets.iterrows():
                if set1['rate_set_id'] >= set2['rate_set_id']:
                    continue

                # Calculate rate difference percentage
                rate_diff = abs(set1['final_rate_amount'] - set2['final_rate_amount'])
                rate_diff_percent = (rate_diff / max(set1['final_rate_amount'], set2['final_rate_amount'])) * 100

                # Check if difference is less than threshold
                if rate_diff_percent >= threshold_percent:
                    continue

                # Check for similar matter/client contexts
                if self._are_contexts_similar(set1, set2):
                    merge_candidates.append({
                        'rate_set_1_id': set1['rate_set_id'],
                        'rate_set_1_name': set1['rate_set_name'],
                        'rate_set_2_id': set2['rate_set_id'],
                        'rate_set_2_name': set2['rate_set_name'],
                        'rate_1': round(set1['final_rate_amount'], 2),
                        'rate_2': round(set2['final_rate_amount'], 2),
                        'rate_difference_percent': round(rate_diff_percent, 2),
                        'practice_areas_1': set1['practice_areas'],
                        'practice_areas_2': set2['practice_areas'],
                        'discount_structure_1': set1['discount_structure'],
                        'discount_structure_2': set2['discount_structure'],
                        'matter_count_1': set1['matter_count'],
                        'matter_count_2': set2['matter_count'],
                        'client_count_1': set1['client_count'],
                        'client_count_2': set2['client_count']
                    })

        logger.info(f"Identified {len(merge_candidates)} merge candidates")
        return pd.DataFrame(merge_candidates)

    def _are_contexts_similar(self, set1, set2):
        """Check if rate sets are assigned to similar matter/client contexts"""
        # Check practice area similarity
        practice_areas_1 = set(str(set1['practice_areas']).split('|')) if set1['practice_areas'] else set()
        practice_areas_2 = set(str(set2['practice_areas']).split('|')) if set2['practice_areas'] else set()

        # Similar if they share practice areas or both have no specific practice areas
        practice_similar = bool(practice_areas_1.intersection(practice_areas_2)) or \
                          (not practice_areas_1 and not practice_areas_2)

        # Check client similarity (if both serve same clients or general clients)
        clients_1 = set(str(set1['assigned_clients']).split('|')) if set1['assigned_clients'] else set()
        clients_2 = set(str(set2['assigned_clients']).split('|')) if set2['assigned_clients'] else set()

        client_similar = bool(clients_1.intersection(clients_2)) or \
                        (not clients_1 and not clients_2)

        return practice_similar or client_similar

    def evaluate_matter_assignments(self, candidates_df):
        """
        Step 3: Evaluate how many matters are assigned to each set—
        if they serve the same practice area with nearly identical rates,
        it's a prime candidate for unification.
        """
        logger.info("Evaluating matter assignments for prime candidates...")

        # Add evaluation criteria
        candidates_df['total_matters'] = candidates_df['matter_count_1'] + candidates_df['matter_count_2']
        candidates_df['same_practice_area'] = candidates_df.apply(
            lambda row: row['practice_areas_1'] == row['practice_areas_2'], axis=1
        )
        candidates_df['nearly_identical_rates'] = candidates_df['rate_difference_percent'] <= 1.0

        # Mark prime candidates
        candidates_df['is_prime_candidate'] = (
            candidates_df['same_practice_area'] &
            candidates_df['nearly_identical_rates'] &
            (candidates_df['total_matters'] > 0)
        )

        prime_candidates = candidates_df[candidates_df['is_prime_candidate']]
        logger.info(f"Found {len(prime_candidates)} prime candidates for unification")

        return candidates_df

    def propose_merges(self, candidates_df):
        """
        Propose merging rate sets into a single rate set if they are mostly redundant.
        For each proposed merge, provide the combined rate set details and justification.
        """
        logger.info("Proposing merges with justifications...")
        proposals = []

        for _, candidate in candidates_df.iterrows():
            # Generate justification
            justification_parts = []

            if candidate['rate_difference_percent'] < 1:
                justification_parts.append("Nearly identical rates")
            else:
                justification_parts.append("Minimal difference in final rates")

            if candidate['practice_areas_1'] == candidate['practice_areas_2']:
                justification_parts.append("Same practice group")
            elif candidate['practice_areas_1'] and candidate['practice_areas_2']:
                justification_parts.append("Similar practice areas")

            if candidate['discount_structure_1'] == candidate['discount_structure_2']:
                justification_parts.append("Similar discount structures")

            justification = "; ".join(justification_parts)

            # Determine primary rate set (one with more matters)
            if candidate['matter_count_1'] >= candidate['matter_count_2']:
                primary_id = candidate['rate_set_1_id']
                primary_name = candidate['rate_set_1_name']
                secondary_id = candidate['rate_set_2_id']
                secondary_name = candidate['rate_set_2_name']
                primary_rate = candidate['rate_1']
                secondary_rate = candidate['rate_2']
            else:
                primary_id = candidate['rate_set_2_id']
                primary_name = candidate['rate_set_2_name']
                secondary_id = candidate['rate_set_1_id']
                secondary_name = candidate['rate_set_1_name']
                primary_rate = candidate['rate_2']
                secondary_rate = candidate['rate_1']

            proposal = {
                'primary_rate_set_id': primary_id,
                'primary_rate_set_name': primary_name,
                'primary_rate': primary_rate,
                'secondary_rate_set_id': secondary_id,
                'secondary_rate_set_name': secondary_name,
                'secondary_rate': secondary_rate,
                'rate_difference_percent': candidate['rate_difference_percent'],
                'justification': justification,
                'is_prime_candidate': candidate.get('is_prime_candidate', False),
                'total_matters_affected': candidate['total_matters'],
                'practice_areas_1': candidate['practice_areas_1'],
                'practice_areas_2': candidate['practice_areas_2'],
                'discount_structure_1': candidate['discount_structure_1'],
                'discount_structure_2': candidate['discount_structure_2'],
                'proposed_unified_name': f"Unified: {primary_name}",
                'proposed_unified_rate': primary_rate
            }

            proposals.append(proposal)

        return pd.DataFrame(proposals)

    def setup_pandasai_agent(self, data):
        """Setup PandasAI agent with the merge proposals data"""
        if not self.llm:
            logger.warning("OpenAI LLM not available - AI features disabled")
            return None

        try:
            self.current_data = data
            self.agent = Agent(data, config={"llm": self.llm, "verbose": True})

            print("🤖 PandasAI Agent ready!")
            print("\n" + "="*60)
            print("📊 DATASET OVERVIEW:")
            print(f"   • Total merge proposals: {len(data)}")

            if 'is_prime_candidate' in data.columns:
                prime_count = len(data[data['is_prime_candidate'] == True])
                print(f"   • Prime candidates: {prime_count}")

            if 'total_matters_affected' in data.columns:
                total_matters = data['total_matters_affected'].sum()
                print(f"   • Total matters affected: {total_matters}")

            print("="*60)
            return self.agent

        except Exception as e:
            logger.error(f"PandasAI setup failed: {e}")
            return None

    def show_basic_summary(self, proposals_df):
        """Show basic summary without generating CSV"""
        print("\n" + "="*80)
        print("RATE SET MERGE PROPOSALS SUMMARY")
        print("="*80)
        print(f"Total merge proposals: {len(proposals_df)}")

        # Prime candidates summary
        prime_proposals = proposals_df[proposals_df['is_prime_candidate'] == True]
        regular_proposals = proposals_df[proposals_df['is_prime_candidate'] == False]

        print(f"Prime candidates: {len(prime_proposals)}")
        print(f"Other candidates: {len(regular_proposals)}")

        if len(proposals_df) > 0:
            print(f"Average rate difference: {proposals_df['rate_difference_percent'].mean():.2f}%")
            print(f"Total matters affected: {proposals_df['total_matters_affected'].sum()}")

            print("\nTop 5 Merge Candidates:")
            top_candidates = proposals_df.nsmallest(5, 'rate_difference_percent')
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                print(f"{i}. {row['primary_rate_set_name']} + {row['secondary_rate_set_name']} "
                      f"({row['rate_difference_percent']}% diff)")

        print("="*80)

    def interactive_analysis(self):
        """Start interactive PandasAI session"""
        if not self.agent or self.current_data is None:
            print("❌ PandasAI agent not available or no data loaded")
            return

        print("\n💡 SAMPLE QUERIES YOU CAN TRY:")
        print("   • 'Show me the top 5 prime candidates'")
        print("   • 'Create a bar chart of merges by rate difference'")
        print("   • 'Show a pie chart of prime vs regular candidates'")
        print("   • 'What is the total savings potential?'")
        print("   • 'Create a scatter plot of rate differences vs matters affected'")
        print("   • 'Show me merges with 0% rate difference'")                                                                                            
        print("   • 'Group merges by practice areas'")

        print("\n" + "="*60)
        print("🔥 INTERACTIVE AI ANALYSIS - Type 'exit' to quit")
        print("="*60)

        while True:
            try:
                user_query = input("\n📝 Your query: ").strip()

                if user_query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break

                if user_query:
                    print(f"\n🤖 Processing: {user_query}")
                    print("⏳ Analyzing data and generating response...")

                    result = self.agent.chat(user_query)
                    print(f"📊 Result: {result}")
                else:
                    print("❓ Please enter a query or 'exit' to quit")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Try a different question or 'exit' to quit")

    def run_merge_detection(self, threshold_percent=5.0):
        """
        Main method that runs the complete merge detection process
        and starts interactive analysis
        """
        try:
            print("🚀 Starting Rate Set Merge Detection...")
            print("Detection Method:")
            print("1. Capturing key attributes (rate amounts, discount structure, matter/client exceptions)")
            print("2. Comparing sets for <5% difference in similar contexts")
            print("3. Evaluating matter assignments for prime candidates")
            print("4. Proposing merges with justifications")
            print("5. Interactive AI analysis with charts") 

            # Step 1: Capture key attributes
            rate_sets = self.capture_key_attributes()

            # Step 2: Compare sets and identify candidates
            candidates = self.compare_sets_and_identify_candidates(rate_sets, threshold_percent)

            if candidates.empty:
                print("No merge candidates found with the specified criteria.")
                return None

            # Step 3: Evaluate matter assignments
            evaluated_candidates = self.evaluate_matter_assignments(candidates)

            # Step 4: Propose merges with justifications
            proposals = self.propose_merges(evaluated_candidates)

            # Step 5: Show basic summary (no CSV)
            self.show_basic_summary(proposals)

            # Step 6: Setup PandasAI for interactive analysis
            if self.setup_pandasai_agent(proposals):
                print("\n🎯 Ready for interactive analysis!")

                # Ask user if they want to start interactive mode
                start_interactive = input("\nStart interactive AI analysis? (y/n): ").strip().lower()
                if start_interactive in ['y', 'yes']:
                    self.interactive_analysis()
            else:
                print("\n📊 Basic analysis completed. AI features not available.")

            return proposals

        except Exception as e:
            logger.error(f"Error in merge detection: {e}")
            return None


def main():
    """Main execution function"""
    merger = RateSetMerger()

    # Run with 5% threshold as specified
    proposals = merger.run_merge_detection(threshold_percent=5.0)

    if proposals is not None:
        print(f"\n✅ Process completed. Found {len(proposals)} merge proposals.")
        print("📊 Data is ready for interactive AI analysis and chart generation!")
    else:
        print("❌ Process completed with no results.")


if __name__ == "__main__":
    main()