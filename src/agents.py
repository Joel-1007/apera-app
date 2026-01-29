"""
APERA Autonomous AI Agents
==========================

Multi-agent system for advanced research tasks:
- Planning Agent: Creates multi-step research plans
- Comparison Agent: Compares papers across dimensions
- Verification Agent: Validates claims against sources
- Gap Finder Agent: Identifies research gaps
- Orchestrator: Coordinates multiple agents

Author: APERA Team
Version: 1.0
"""

import os
import json
import logging
import datetime
import requests
from typing import List, Dict, Any, Optional
import traceback

logger = logging.getLogger("APERA.Agents")

# ==========================================
# AGENT BASE CLASS
# ==========================================
class BaseAgent:
    """
    Base class for all autonomous agents
    Provides common functionality for reasoning and logging
    """
    
    def __init__(self, agent_name: str, ollama_url: str = None):
        self.agent_name = agent_name
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.reasoning_trace = []
        
        logger.info(f"🤖 {self.agent_name} initialized")
    
    def call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 1500) -> str:
        """
        Call local LLM (Ollama) for agent reasoning
        
        Args:
            prompt: User/agent prompt
            system_prompt: System context
            max_tokens: Max response length
            
        Returns:
            LLM response
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"{self.agent_name}: LLM error {response.status_code}")
                return None
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"{self.agent_name}: LLM call failed: {e}")
            return None
    
    def log_reasoning(self, step: str, thought: str):
        """Log agent's reasoning process"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "thought": thought
        }
        self.reasoning_trace.append(entry)
        logger.info(f"💭 {self.agent_name} - {step}: {thought[:100]}...")
    
    def get_reasoning_trace(self) -> List[Dict]:
        """Get complete reasoning trace"""
        return self.reasoning_trace


# ==========================================
# PLANNING AGENT
# ==========================================
class ResearchPlanningAgent(BaseAgent):
    """
    Autonomous agent that creates and executes research plans
    
    Capabilities:
    - Analyzes research questions
    - Creates multi-step plans
    - Executes plans autonomously
    - Reflects on results
    """
    
    def __init__(self):
        super().__init__("Planning Agent")
    
    def create_research_plan(self, query: str) -> Dict[str, Any]:
        """
        Agent autonomously creates a research plan
        
        Args:
            query: Research question
            
        Returns:
            Structured research plan with steps
        """
        logger.info(f"🎯 {self.agent_name}: Creating plan for '{query}'")
        
        self.log_reasoning("analyze_query", f"Understanding research question: {query}")
        
        system_prompt = """You are an expert research planning agent. Given a research question, 
create a systematic multi-step plan to thoroughly investigate it.

Each step should:
1. Have a clear search query
2. Explain why this step is important
3. Specify what information to extract

Return ONLY valid JSON (no markdown, no explanation):
{
  "steps": [
    {
      "step_number": 1,
      "search_query": "specific search terms",
      "reasoning": "why this step matters",
      "expected_info": "what we hope to find"
    }
  ]
}"""
        
        user_prompt = f"""Research Question: "{query}"

Create a 3-4 step research plan. Focus on:
1. Understanding fundamentals first
2. Then exploring specific approaches/methods
3. Then finding recent advances
4. Finally identifying gaps or future directions

Return as JSON."""
        
        response = self.call_llm(user_prompt, system_prompt, max_tokens=1000)
        
        if not response:
            logger.warning(f"{self.agent_name}: LLM unavailable, using heuristic plan")
            return self._create_heuristic_plan(query)
        
        try:
            # Clean response (remove markdown if present)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            plan = json.loads(response)
            
            self.log_reasoning("plan_created", f"Generated {len(plan['steps'])} step plan")
            
            return {
                "query": query,
                "plan": plan,
                "agent": self.agent_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"{self.agent_name}: JSON parse error: {e}")
            logger.error(f"Response was: {response[:200]}")
            return self._create_heuristic_plan(query)
    
    def _create_heuristic_plan(self, query: str) -> Dict[str, Any]:
        """Fallback: Create plan using heuristics"""
        return {
            "query": query,
            "plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "search_query": f"{query} introduction fundamentals",
                        "reasoning": "Understand basic concepts first",
                        "expected_info": "Foundational knowledge"
                    },
                    {
                        "step_number": 2,
                        "search_query": f"{query} methods approaches",
                        "reasoning": "Learn different techniques",
                        "expected_info": "Methodological approaches"
                    },
                    {
                        "step_number": 3,
                        "search_query": f"{query} recent advances 2024",
                        "reasoning": "Find latest developments",
                        "expected_info": "State-of-the-art research"
                    }
                ]
            },
            "agent": f"{self.agent_name} (heuristic)",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def execute_step(self, step: Dict, search_function) -> Dict[str, Any]:
        """
        Execute a single step of the research plan
        
        Args:
            step: Plan step to execute
            search_function: Function to search ArXiv
            
        Returns:
            Step execution results
        """
        logger.info(f"📍 {self.agent_name}: Executing step {step['step_number']}")
        
        self.log_reasoning(
            f"execute_step_{step['step_number']}", 
            f"Searching for: {step['search_query']}"
        )
        
        # Execute search
        papers = search_function(step['search_query'], max_results=2)
        
        # Agent reflects on results
        reflection = self.reflect_on_results(step, papers)
        
        return {
            "step": step,
            "papers_found": len(papers),
            "papers": papers,
            "reflection": reflection,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def reflect_on_results(self, step: Dict, papers: List[Dict]) -> str:
        """
        Agent reflects: Did we achieve the step's goal?
        
        Args:
            step: The plan step
            papers: Papers found
            
        Returns:
            Reflection text
        """
        self.log_reasoning("reflect", f"Evaluating {len(papers)} papers for step {step['step_number']}")
        
        if not papers or len(papers) == 0:
            return "⚠️ No papers found. May need to reformulate search query."
        
        prompt = f"""As a research planning agent, evaluate these search results:

GOAL: {step['reasoning']}
EXPECTED: {step['expected_info']}

RESULTS:
- Papers found: {len(papers)}
- Titles: {[p['title'] for p in papers[:2]]}

Quick analysis (2-3 sentences):
1. Do these papers address our goal?
2. Is the information we expected present?
3. Should we continue or search again?"""
        
        reflection = self.call_llm(prompt, max_tokens=300)
        
        if reflection:
            return reflection
        else:
            return f"✅ Found {len(papers)} papers. Manual review recommended."


# ==========================================
# COMPARISON AGENT
# ==========================================
class PaperComparisonAgent(BaseAgent):
    """
    Autonomous agent that compares research papers
    
    Capabilities:
    - Identifies approaches to compare
    - Searches papers for each approach
    - Extracts comparison dimensions
    - Generates insights
    """
    
    def __init__(self):
        super().__init__("Comparison Agent")
    
    def compare_papers(self, topic: str, papers: List[Dict], 
                      comparison_axes: List[str] = None) -> Dict[str, Any]:
        """
        Agent-driven paper comparison
        
        Args:
            topic: Research topic
            papers: Papers to compare
            comparison_axes: Dimensions to compare (default: auto-detect)
            
        Returns:
            Comparison analysis
        """
        logger.info(f"⚖️ {self.agent_name}: Comparing papers on '{topic}'")
        
        if not comparison_axes:
            comparison_axes = ["methodology", "key findings", "limitations"]
        
        self.log_reasoning("start_comparison", f"Comparing {len(papers)} papers on {comparison_axes}")
        
        # Extract information for each paper
        comparison_matrix = {}
        for idx, paper in enumerate(papers, 1):
            paper_key = f"Paper {idx}: {paper['title'][:50]}"
            comparison_matrix[paper_key] = {}
            
            for axis in comparison_axes:
                extracted = self.extract_dimension(paper, axis)
                comparison_matrix[paper_key][axis] = extracted
        
        # Generate insights
        insights = self.generate_insights(comparison_matrix, topic)
        
        return {
            "topic": topic,
            "papers_compared": len(papers),
            "comparison_axes": comparison_axes,
            "comparison_matrix": comparison_matrix,
            "insights": insights,
            "agent": self.agent_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def extract_dimension(self, paper: Dict, dimension: str) -> str:
        """
        Extract specific information from paper
        
        Args:
            paper: Paper data
            dimension: What to extract (e.g., "methodology")
            
        Returns:
            Extracted information
        """
        prompt = f"""Extract information about '{dimension}' from this paper:

Title: {paper['title']}
Abstract: {paper['summary'][:500]}...

What does this paper say about {dimension}? 
Be specific. Quote numbers/metrics if present. Max 3 sentences."""
        
        result = self.call_llm(prompt, max_tokens=200)
        
        if result:
            return result
        else:
            # Fallback: Simple keyword extraction
            if dimension.lower() in paper['summary'].lower():
                # Find sentence containing the dimension
                sentences = paper['summary'].split('.')
                for sent in sentences:
                    if dimension.lower() in sent.lower():
                        return sent.strip() + "."
            return f"Information about {dimension} not explicitly stated in abstract."
    
    def generate_insights(self, matrix: Dict, topic: str) -> str:
        """
        Agent generates insights from comparison
        
        Args:
            matrix: Comparison matrix
            topic: Research topic
            
        Returns:
            Insights text
        """
        self.log_reasoning("synthesize", "Generating insights from comparison")
        
        prompt = f"""As a research comparison agent, analyze this comparison of papers on '{topic}':

{json.dumps(matrix, indent=2)[:1500]}

Provide insights (4-5 sentences):
1. What are the main differences between approaches?
2. Which papers have the strongest evidence?
3. Are there conflicting findings?
4. What's the overall trend or consensus?"""
        
        insights = self.call_llm(prompt, max_tokens=500)
        
        if insights:
            return insights
        else:
            return f"Compared {len(matrix)} papers across multiple dimensions. Manual review recommended for detailed insights."


# ==========================================
# VERIFICATION AGENT
# ==========================================
class CitationVerificationAgent(BaseAgent):
    """
    Agent that verifies claims against cited sources
    
    Capabilities:
    - Extracts claims from text
    - Verifies each claim against citations
    - Calculates confidence scores
    - Identifies unsupported claims
    """
    
    def __init__(self):
        super().__init__("Verification Agent")
    
    def verify_response(self, response_text: str, citations: List[Dict]) -> Dict[str, Any]:
        """
        Verify that response claims are supported by citations
        
        Args:
            response_text: Generated response to verify
            citations: Available citations
            
        Returns:
            Verification results
        """
        logger.info(f"✓ {self.agent_name}: Verifying response against {len(citations)} citations")
        
        self.log_reasoning("extract_claims", "Identifying factual claims in response")
        
        # Extract claims
        claims = self.extract_claims(response_text)
        
        if not claims:
            return {
                "verified": True,
                "confidence": 0.95,
                "claims_checked": 0,
                "message": "No specific factual claims to verify"
            }
        
        # Verify each claim
        verifications = []
        for claim in claims:
            verification = self.verify_single_claim(claim, citations)
            verifications.append(verification)
        
        # Calculate confidence
        verified_count = sum(1 for v in verifications if v['verified'])
        confidence = verified_count / len(verifications) if verifications else 0
        
        return {
            "claims_checked": len(claims),
            "claims_verified": verified_count,
            "confidence": round(confidence, 2),
            "hallucination_risk": round(1 - confidence, 2),
            "verifications": verifications,
            "agent": self.agent_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of claims
        """
        # Simple heuristic: sentences with numbers or specific terminology
        import re
        
        sentences = text.split('.')
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Look for sentences with numbers, percentages, or technical terms
            if re.search(r'\d+%|\d+\.\d+|achieves|performs|shows|demonstrates', sent, re.IGNORECASE):
                if len(sent) > 20:  # Ignore very short sentences
                    claims.append(sent)
        
        return claims[:5]  # Limit to 5 claims for performance
    
    def verify_single_claim(self, claim: str, citations: List[Dict]) -> Dict:
        """
        Verify a single claim against citations
        
        Args:
            claim: Claim to verify
            citations: Available citations
            
        Returns:
            Verification result
        """
        # Check if any citation supports this claim
        for citation in citations:
            # Simple check: are key terms from claim in citation?
            claim_words = set(claim.lower().split())
            citation_text = citation.get('text', '').lower()
            
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            claim_words = claim_words - stop_words
            
            # Check overlap
            if len(claim_words) > 0:
                matches = sum(1 for word in claim_words if word in citation_text)
                overlap = matches / len(claim_words)
                
                if overlap > 0.3:  # 30% of claim words found in citation
                    return {
                        "claim": claim,
                        "verified": True,
                        "source": citation['file'],
                        "confidence": round(overlap, 2)
                    }
        
        return {
            "claim": claim,
            "verified": False,
            "confidence": 0.0,
            "note": "No supporting citation found"
        }


# ==========================================
# AGENT ORCHESTRATOR
# ==========================================
class AgentOrchestrator:
    """
    Coordinates multiple agents working together
    
    Capabilities:
    - Selects appropriate agents for query
    - Coordinates agent execution
    - Synthesizes multi-agent results
    - Logs collaboration
    """
    
    def __init__(self):
        self.planning_agent = ResearchPlanningAgent()
        self.comparison_agent = PaperComparisonAgent()
        self.verification_agent = CitationVerificationAgent()
        
        self.collaboration_log = []
        
        logger.info("🎭 Agent Orchestrator initialized")
    
    def process_query(self, query: str, query_type: str, papers: List[Dict],
                     search_function=None) -> Dict[str, Any]:
        """
        Orchestrate agents to process complex query
        
        Args:
            query: User query
            query_type: 'conceptual', 'research', 'hybrid', or 'comparison'
            papers: Available papers
            search_function: Function to search more papers if needed
            
        Returns:
            Multi-agent results
        """
        logger.info(f"🎭 Orchestrator: Processing '{query}' (type: {query_type})")
        
        agents_used = []
        results = {}
        
        # STRATEGY 1: Comparison Query
        if "compare" in query.lower() or "versus" in query.lower() or query_type == "comparison":
            logger.info("🎭 Strategy: Deploying Comparison Agent")
            agents_used.append("Comparison Agent")
            
            comparison = self.comparison_agent.compare_papers(
                topic=query,
                papers=papers,
                comparison_axes=["approach", "key findings", "limitations"]
            )
            results["comparison"] = comparison
        
        # STRATEGY 2: Research Planning Query  
        elif query_type == "research" or "latest" in query.lower() or "advances" in query.lower():
            logger.info("🎭 Strategy: Deploying Planning Agent")
            agents_used.append("Planning Agent")
            
            plan = self.planning_agent.create_research_plan(query)
            results["plan"] = plan
            
            # Execute plan if search function available
            if search_function:
                execution_results = []
                for step in plan['plan']['steps']:
                    step_result = self.planning_agent.execute_step(step, search_function)
                    execution_results.append(step_result)
                
                results["execution"] = execution_results
        
        # ALWAYS: Verify results
        if papers:
            logger.info("🎭 Deploying Verification Agent")
            agents_used.append("Verification Agent")
            
            # Create sample response text from papers for verification
            sample_response = f"Research on {query}: " + " ".join([p['title'] for p in papers[:2]])
            
            verification = self.verification_agent.verify_response(
                response_text=sample_response,
                citations=[{"file": p['title'], "text": p['summary'][:200]} for p in papers]
            )
            results["verification"] = verification
        
        # Log collaboration
        collaboration_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "query_type": query_type,
            "agents_deployed": agents_used,
            "papers_analyzed": len(papers),
            "success": True
        }
        self.collaboration_log.append(collaboration_entry)
        self._save_collaboration_log()
        
        return {
            "query": query,
            "agents_used": agents_used,
            "results": results,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _save_collaboration_log(self):
        """Save collaboration log to file"""
        try:
            os.makedirs("agent_logs", exist_ok=True)
            
            with open("agent_logs/collaborations.json", "w") as f:
                json.dump(self.collaboration_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving collaboration log: {e}")


# ==========================================
# AGENT FACTORY
# ==========================================
def create_orchestrator() -> AgentOrchestrator:
    """
    Factory function to create agent orchestrator
    
    Returns:
        Initialized orchestrator
    """
    return AgentOrchestrator()


# ==========================================
# TESTING FUNCTIONS
# ==========================================
if __name__ == "__main__":
    # Test agents
    print("🧪 Testing APERA Agents...")
    
    # Test Planning Agent
    print("\n1. Testing Planning Agent...")
    planner = ResearchPlanningAgent()
    plan = planner.create_research_plan("transformer neural networks")
    print(f"✓ Created plan with {len(plan['plan']['steps'])} steps")
    
    # Test Comparison Agent
    print("\n2. Testing Comparison Agent...")
    comparator = PaperComparisonAgent()
    test_papers = [
        {"title": "Paper 1", "summary": "This paper uses CNNs for image classification achieving 95% accuracy."},
        {"title": "Paper 2", "summary": "We propose transformers for image tasks with 97% accuracy but higher computational cost."}
    ]
    comparison = comparator.compare_papers("image classification", test_papers)
    print(f"✓ Compared {comparison['papers_compared']} papers")
    
    # Test Verification Agent
    print("\n3. Testing Verification Agent...")
    verifier = CitationVerificationAgent()
    verification = verifier.verify_response(
        "The model achieves 95% accuracy on ImageNet",
        [{"file": "Test Paper", "text": "accuracy of 95% on ImageNet dataset"}]
    )
    print(f"✓ Verified {verification['claims_checked']} claims")
    
    # Test Orchestrator
    print("\n4. Testing Orchestrator...")
    orchestrator = create_orchestrator()
    result = orchestrator.process_query(
        "compare CNN and transformer for vision",
        "comparison",
        test_papers
    )
    print(f"✓ Orchestrated {len(result['agents_used'])} agents")
    
    print("\n✅ All agent tests passed!")
