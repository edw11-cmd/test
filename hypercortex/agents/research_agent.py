"""
Research Agent - Information gathering and analysis
"""

import json
from typing import Dict, List, Any, Optional
import structlog

from .base_agent import BaseAgent, AgentAction, AgentObservation
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class ResearchAgent(BaseAgent):
    """Agent specialized in research, information gathering, and analysis"""
    
    def __init__(self):
        system_prompt = """You are a research specialist in the HyperCortex-AI system. Your role is to:

1. Gather comprehensive information on topics and questions
2. Analyze and synthesize information from multiple sources
3. Identify key insights, trends, and patterns
4. Provide evidence-based conclusions and recommendations
5. Fact-check and verify information accuracy
6. Create structured research reports and summaries

You approach research systematically and critically evaluate sources for credibility and relevance.
You always provide citations and acknowledge limitations in your findings.

When conducting research:
- Start with a clear research question or objective
- Use multiple sources and perspectives
- Evaluate source credibility and bias
- Look for patterns and contradictions
- Synthesize findings into actionable insights
- Acknowledge uncertainties and limitations
- Provide clear, well-structured reports

Research areas include:
- Technology trends and developments
- Market analysis and competitive intelligence
- Academic and scientific literature
- Best practices and methodologies
- Regulatory and compliance information
- Industry standards and frameworks"""

        super().__init__(
            name="ResearchAgent",
            role="Research Specialist",
            system_prompt=system_prompt
        )
    
    async def _decide_next_action(self, task: str) -> Optional[AgentAction]:
        """Decide the next research action"""
        
        # Check if we need to define research scope
        scope_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "define_research_scope"
        ]
        
        if not scope_actions:
            return AgentAction(
                action_type="define_research_scope",
                parameters={"task": task},
                reasoning="Need to define the research scope and objectives",
                confidence=0.9
            )
        
        # Check if we need to gather information
        gather_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "gather_information"
        ]
        
        if not gather_actions:
            return AgentAction(
                action_type="gather_information",
                parameters={"task": task},
                reasoning="Need to gather relevant information",
                confidence=0.8
            )
        
        # Check if we need to analyze findings
        analysis_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "analyze_findings"
        ]
        
        if not analysis_actions and gather_actions:
            return AgentAction(
                action_type="analyze_findings",
                parameters={"task": task},
                reasoning="Need to analyze gathered information",
                confidence=0.8
            )
        
        # Check if we need to synthesize results
        synthesis_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "synthesize_results"
        ]
        
        if not synthesis_actions and analysis_actions:
            return AgentAction(
                action_type="synthesize_results",
                parameters={"task": task},
                reasoning="Need to synthesize research findings",
                confidence=0.8
            )
        
        # Check if we need to validate findings
        validation_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "validate_findings"
        ]
        
        if not validation_actions and synthesis_actions:
            return AgentAction(
                action_type="validate_findings",
                parameters={"task": task},
                reasoning="Need to validate research findings",
                confidence=0.7
            )
        
        # Research is complete
        return None
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute research actions"""
        
        if action.action_type == "define_research_scope":
            return await self._define_research_scope(action.parameters["task"])
        
        elif action.action_type == "gather_information":
            return await self._gather_information(action.parameters["task"])
        
        elif action.action_type == "analyze_findings":
            return await self._analyze_findings(action.parameters["task"])
        
        elif action.action_type == "synthesize_results":
            return await self._synthesize_results(action.parameters["task"])
        
        elif action.action_type == "validate_findings":
            return await self._validate_findings(action.parameters["task"])
        
        else:
            return AgentObservation(
                content=f"Unknown action type: {action.action_type}",
                success=False,
                metadata={"action_type": action.action_type}
            )
    
    async def _is_task_complete(self, task: str, last_observation: AgentObservation) -> bool:
        """Check if research task is complete"""
        
        # Research is complete when we have validated findings
        validation_actions = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "validate_findings" and obs.success
        ]
        
        return len(validation_actions) > 0
    
    async def _define_research_scope(self, task: str) -> AgentObservation:
        """Define the research scope and objectives"""
        
        scope_prompt = f"""
Research task: {task}

Define a comprehensive research scope for this task:

1. **Research Objectives**:
   - Primary research question(s)
   - Secondary questions to explore
   - Expected outcomes and deliverables

2. **Scope Definition**:
   - What topics to include/exclude
   - Time frame for information (if relevant)
   - Geographic or domain limitations
   - Depth of research required

3. **Information Sources**:
   - Types of sources to consult
   - Preferred databases or repositories
   - Expert opinions needed
   - Primary vs secondary sources

4. **Research Methodology**:
   - Approach to information gathering
   - Analysis methods to use
   - Quality criteria for sources
   - Validation strategies

5. **Success Criteria**:
   - How to measure research completeness
   - Quality standards for findings
   - Confidence levels required

Provide the research scope in JSON format:
{{
    "research_objectives": {{
        "primary_question": "main research question",
        "secondary_questions": ["question1", "question2"],
        "expected_outcomes": ["outcome1", "outcome2"]
    }},
    "scope": {{
        "included_topics": ["topic1", "topic2"],
        "excluded_topics": ["topic1", "topic2"],
        "time_frame": "time period",
        "domain_focus": "specific domain",
        "research_depth": "shallow|moderate|deep"
    }},
    "information_sources": {{
        "source_types": ["academic", "industry", "news"],
        "preferred_databases": ["db1", "db2"],
        "expert_input": "requirements",
        "primary_sources": true
    }},
    "methodology": {{
        "gathering_approach": "systematic|exploratory|targeted",
        "analysis_methods": ["method1", "method2"],
        "quality_criteria": ["criteria1", "criteria2"],
        "validation_strategy": "approach"
    }},
    "success_criteria": {{
        "completeness_measures": ["measure1", "measure2"],
        "quality_standards": ["standard1", "standard2"],
        "confidence_threshold": 0.8
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=scope_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "define_research_scope"}
            )
            
            # Try to parse as JSON
            try:
                scope_data = json.loads(response.content)
                
                # Store research scope in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "research_scope",
                        "task": task,
                        "research_depth": scope_data.get("scope", {}).get("research_depth", "moderate")
                    },
                    category="research_scopes"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "define_research_scope",
                        "research_depth": scope_data.get("scope", {}).get("research_depth", "moderate"),
                        "source_types": len(scope_data.get("information_sources", {}).get("source_types", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Research scope defined but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "define_research_scope", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error defining research scope: {str(e)}")
            return AgentObservation(
                content=f"Error defining research scope: {str(e)}",
                success=False,
                metadata={"action_type": "define_research_scope", "error": str(e)}
            )
    
    async def _gather_information(self, task: str) -> AgentObservation:
        """Gather information based on research scope"""
        
        # Get research scope from previous step
        scope_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "define_research_scope" and obs.success
        ]
        
        scope = scope_obs[-1].content if scope_obs else "No research scope available"
        
        gather_prompt = f"""
Research task: {task}

Research scope:
{scope}

Based on the defined scope, gather comprehensive information on the research topic.
Since I cannot access external sources directly, I will use my knowledge base to provide relevant information.

Provide detailed information covering:

1. **Core Concepts and Definitions**:
   - Key terms and concepts
   - Fundamental principles
   - Important distinctions and classifications

2. **Current State and Trends**:
   - Current status of the topic
   - Recent developments and changes
   - Emerging trends and patterns

3. **Key Players and Stakeholders**:
   - Important organizations, companies, or individuals
   - Their roles and contributions
   - Relationships and interactions

4. **Best Practices and Standards**:
   - Industry standards and guidelines
   - Proven methodologies and approaches
   - Common pitfalls and how to avoid them

5. **Challenges and Opportunities**:
   - Current challenges and limitations
   - Potential solutions and innovations
   - Future opportunities and directions

6. **Supporting Evidence**:
   - Relevant data and statistics
   - Case studies and examples
   - Research findings and conclusions

Format the information as a structured research report:
{{
    "core_concepts": {{
        "definitions": {{"term1": "definition1"}},
        "principles": ["principle1", "principle2"],
        "classifications": ["type1", "type2"]
    }},
    "current_state": {{
        "status": "description",
        "recent_developments": ["dev1", "dev2"],
        "trends": ["trend1", "trend2"]
    }},
    "key_players": [
        {{
            "name": "player_name",
            "role": "description",
            "contribution": "what they do"
        }}
    ],
    "best_practices": {{
        "standards": ["standard1", "standard2"],
        "methodologies": ["method1", "method2"],
        "pitfalls": ["pitfall1", "pitfall2"]
    }},
    "challenges_opportunities": {{
        "challenges": ["challenge1", "challenge2"],
        "solutions": ["solution1", "solution2"],
        "opportunities": ["opp1", "opp2"]
    }},
    "evidence": {{
        "statistics": ["stat1", "stat2"],
        "case_studies": ["case1", "case2"],
        "research_findings": ["finding1", "finding2"]
    }},
    "information_quality": {{
        "confidence_level": 0.8,
        "source_diversity": "high|medium|low",
        "recency": "current|recent|dated"
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=gather_prompt,
                system_message=self.system_prompt,
                max_tokens=4000,  # Allow for comprehensive information
                metadata={"agent": self.name, "action": "gather_information"}
            )
            
            # Try to parse as JSON
            try:
                info_data = json.loads(response.content)
                
                # Store gathered information in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "research_information",
                        "task": task,
                        "confidence_level": info_data.get("information_quality", {}).get("confidence_level", 0.7)
                    },
                    category="research_data"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "gather_information",
                        "confidence_level": info_data.get("information_quality", {}).get("confidence_level", 0.7),
                        "key_players_count": len(info_data.get("key_players", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Information gathered but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "gather_information", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error gathering information: {str(e)}")
            return AgentObservation(
                content=f"Error gathering information: {str(e)}",
                success=False,
                metadata={"action_type": "gather_information", "error": str(e)}
            )
    
    async def _analyze_findings(self, task: str) -> AgentObservation:
        """Analyze the gathered information"""
        
        # Get gathered information from previous step
        info_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "gather_information" and obs.success
        ]
        
        if not info_obs:
            return AgentObservation(
                content="No information found to analyze",
                success=False,
                metadata={"action_type": "analyze_findings", "error": "no_information"}
            )
        
        information = info_obs[-1].content
        
        analysis_prompt = f"""
Research task: {task}

Gathered information:
{information}

Analyze the gathered information to extract insights and patterns:

1. **Key Insights**:
   - Most important findings
   - Surprising or unexpected discoveries
   - Critical success factors

2. **Pattern Analysis**:
   - Common themes and patterns
   - Relationships between concepts
   - Cause-and-effect relationships

3. **Gap Analysis**:
   - Information gaps identified
   - Areas needing further research
   - Conflicting information found

4. **Trend Analysis**:
   - Emerging trends and directions
   - Historical patterns and evolution
   - Future implications

5. **Risk Assessment**:
   - Potential risks and threats
   - Mitigation strategies
   - Uncertainty factors

6. **Opportunity Identification**:
   - Market opportunities
   - Innovation possibilities
   - Competitive advantages

Provide the analysis in JSON format:
{{
    "key_insights": [
        {{
            "insight": "description",
            "importance": "high|medium|low",
            "evidence": "supporting evidence"
        }}
    ],
    "patterns": [
        {{
            "pattern": "description",
            "frequency": "common|occasional|rare",
            "implications": "what it means"
        }}
    ],
    "gaps": {{
        "information_gaps": ["gap1", "gap2"],
        "research_needs": ["need1", "need2"],
        "conflicts": ["conflict1", "conflict2"]
    }},
    "trends": [
        {{
            "trend": "description",
            "direction": "increasing|decreasing|stable",
            "timeline": "short|medium|long term",
            "impact": "high|medium|low"
        }}
    ],
    "risks": [
        {{
            "risk": "description",
            "probability": "high|medium|low",
            "impact": "high|medium|low",
            "mitigation": "strategy"
        }}
    ],
    "opportunities": [
        {{
            "opportunity": "description",
            "potential": "high|medium|low",
            "requirements": ["req1", "req2"],
            "timeline": "timeframe"
        }}
    ],
    "analysis_quality": {{
        "confidence": 0.8,
        "completeness": "high|medium|low",
        "reliability": "high|medium|low"
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=analysis_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "analyze_findings"}
            )
            
            # Try to parse as JSON
            try:
                analysis_data = json.loads(response.content)
                
                # Store analysis in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "research_analysis",
                        "task": task,
                        "confidence": analysis_data.get("analysis_quality", {}).get("confidence", 0.7)
                    },
                    category="research_analysis"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "analyze_findings",
                        "confidence": analysis_data.get("analysis_quality", {}).get("confidence", 0.7),
                        "insights_count": len(analysis_data.get("key_insights", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Analysis completed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "analyze_findings", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error analyzing findings: {str(e)}")
            return AgentObservation(
                content=f"Error analyzing findings: {str(e)}",
                success=False,
                metadata={"action_type": "analyze_findings", "error": str(e)}
            )
    
    async def _synthesize_results(self, task: str) -> AgentObservation:
        """Synthesize research results into actionable conclusions"""
        
        # Get analysis from previous step
        analysis_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "analyze_findings" and obs.success
        ]
        
        if not analysis_obs:
            return AgentObservation(
                content="No analysis found to synthesize",
                success=False,
                metadata={"action_type": "synthesize_results", "error": "no_analysis"}
            )
        
        analysis = analysis_obs[-1].content
        
        synthesis_prompt = f"""
Research task: {task}

Analysis results:
{analysis}

Synthesize the research findings into actionable conclusions and recommendations:

1. **Executive Summary**:
   - Key findings in 2-3 sentences
   - Main conclusions
   - Overall assessment

2. **Strategic Recommendations**:
   - Specific actions to take
   - Priority levels
   - Implementation considerations

3. **Decision Support**:
   - Factors to consider in decision making
   - Trade-offs and alternatives
   - Success metrics

4. **Implementation Roadmap**:
   - Immediate next steps
   - Medium-term actions
   - Long-term considerations

5. **Risk Mitigation**:
   - Key risks to monitor
   - Contingency plans
   - Early warning indicators

6. **Future Research**:
   - Areas for further investigation
   - Monitoring requirements
   - Update triggers

Provide the synthesis in JSON format:
{{
    "executive_summary": {{
        "key_findings": "summary",
        "main_conclusions": ["conclusion1", "conclusion2"],
        "overall_assessment": "positive|neutral|negative"
    }},
    "recommendations": [
        {{
            "recommendation": "description",
            "priority": "high|medium|low",
            "rationale": "why this is recommended",
            "implementation": "how to implement",
            "timeline": "timeframe",
            "resources_needed": ["resource1", "resource2"]
        }}
    ],
    "decision_support": {{
        "key_factors": ["factor1", "factor2"],
        "alternatives": ["alt1", "alt2"],
        "success_metrics": ["metric1", "metric2"]
    }},
    "implementation_roadmap": {{
        "immediate_actions": ["action1", "action2"],
        "medium_term": ["action1", "action2"],
        "long_term": ["action1", "action2"]
    }},
    "risk_mitigation": {{
        "key_risks": ["risk1", "risk2"],
        "contingencies": ["plan1", "plan2"],
        "monitoring": ["indicator1", "indicator2"]
    }},
    "future_research": {{
        "investigation_areas": ["area1", "area2"],
        "monitoring_needs": ["need1", "need2"],
        "update_triggers": ["trigger1", "trigger2"]
    }},
    "synthesis_quality": {{
        "actionability": "high|medium|low",
        "completeness": "high|medium|low",
        "confidence": 0.85
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=synthesis_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "synthesize_results"}
            )
            
            # Try to parse as JSON
            try:
                synthesis_data = json.loads(response.content)
                
                # Store synthesis in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "research_synthesis",
                        "task": task,
                        "confidence": synthesis_data.get("synthesis_quality", {}).get("confidence", 0.7)
                    },
                    category="research_results"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "synthesize_results",
                        "confidence": synthesis_data.get("synthesis_quality", {}).get("confidence", 0.7),
                        "recommendations_count": len(synthesis_data.get("recommendations", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Synthesis completed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "synthesize_results", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error synthesizing results: {str(e)}")
            return AgentObservation(
                content=f"Error synthesizing results: {str(e)}",
                success=False,
                metadata={"action_type": "synthesize_results", "error": str(e)}
            )
    
    async def _validate_findings(self, task: str) -> AgentObservation:
        """Validate research findings and conclusions"""
        
        # Get synthesis from previous step
        synthesis_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "synthesize_results" and obs.success
        ]
        
        if not synthesis_obs:
            return AgentObservation(
                content="No synthesis found to validate",
                success=False,
                metadata={"action_type": "validate_findings", "error": "no_synthesis"}
            )
        
        synthesis = synthesis_obs[-1].content
        
        validation_prompt = f"""
Research task: {task}

Research synthesis to validate:
{synthesis}

Validate the research findings and conclusions by checking:

1. **Logical Consistency**:
   - Are conclusions logically derived from findings?
   - Are there any contradictions?
   - Do recommendations align with analysis?

2. **Evidence Quality**:
   - Is the evidence sufficient to support conclusions?
   - Are there gaps in supporting evidence?
   - How reliable are the information sources?

3. **Bias Assessment**:
   - Are there potential biases in the research?
   - Are multiple perspectives considered?
   - Are assumptions clearly stated?

4. **Completeness Check**:
   - Are all important aspects covered?
   - Are there missing considerations?
   - Is the scope appropriate?

5. **Actionability Review**:
   - Are recommendations specific and actionable?
   - Are implementation considerations realistic?
   - Are success metrics appropriate?

6. **Risk Assessment**:
   - Are potential risks adequately identified?
   - Are mitigation strategies realistic?
   - Are uncertainties acknowledged?

Provide validation results in JSON format:
{{
    "validation_results": {{
        "logical_consistency": {{
            "score": 0.85,
            "issues": ["issue1", "issue2"],
            "assessment": "strong|adequate|weak"
        }},
        "evidence_quality": {{
            "score": 0.80,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "assessment": "strong|adequate|weak"
        }},
        "bias_assessment": {{
            "score": 0.75,
            "potential_biases": ["bias1", "bias2"],
            "mitigation": ["action1", "action2"],
            "assessment": "low|medium|high bias"
        }},
        "completeness": {{
            "score": 0.90,
            "covered_aspects": ["aspect1", "aspect2"],
            "missing_aspects": ["aspect1", "aspect2"],
            "assessment": "comprehensive|adequate|limited"
        }},
        "actionability": {{
            "score": 0.85,
            "actionable_items": 5,
            "implementation_clarity": "clear|moderate|unclear",
            "assessment": "highly actionable|moderately actionable|limited actionability"
        }},
        "risk_coverage": {{
            "score": 0.80,
            "identified_risks": 3,
            "mitigation_quality": "strong|adequate|weak",
            "assessment": "comprehensive|adequate|limited"
        }}
    }},
    "overall_validation": {{
        "overall_score": 0.82,
        "validation_status": "validated|needs_revision|rejected",
        "confidence_level": "high|medium|low",
        "key_strengths": ["strength1", "strength2"],
        "areas_for_improvement": ["area1", "area2"],
        "recommendations_for_improvement": ["rec1", "rec2"]
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=validation_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "validate_findings"}
            )
            
            # Try to parse as JSON
            try:
                validation_data = json.loads(response.content)
                is_validated = validation_data.get("overall_validation", {}).get("validation_status") == "validated"
                
                return AgentObservation(
                    content=response.content,
                    success=is_validated,
                    metadata={
                        "action_type": "validate_findings",
                        "validation_status": validation_data.get("overall_validation", {}).get("validation_status"),
                        "overall_score": validation_data.get("overall_validation", {}).get("overall_score", 0.0)
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Validation completed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "validate_findings", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error validating findings: {str(e)}")
            return AgentObservation(
                content=f"Error validating findings: {str(e)}",
                success=False,
                metadata={"action_type": "validate_findings", "error": str(e)}
            )