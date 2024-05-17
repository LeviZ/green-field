import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

load_dotenv()


@CrewBase
class ProblemSolvingCrew:
    """Problem Solving Crew"""
    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'
    use_local_llm = os.getenv("USE_LOCAL_LLM", "False") == "True"

    def __init__(self) -> None:
        # Ensure the paths are set correctly
        self.agents_config_path = os.path.join(os.path.dirname(__file__), self.agents_config_path)
        self.tasks_config_path = os.path.join(os.path.dirname(__file__), self.tasks_config_path)

        if self.use_local_llm:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name=os.getenv("OPENAI_MODEL_NAME"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                temperature=1,
                model_name="llama3-70b-8192",
                api_key=os.getenv("GROQ_API_KEY")
            )

        self.agents_config = self.load_yaml_config(self.agents_config_path)
        self.tasks_config = self.load_yaml_config(self.tasks_config_path)

    def load_yaml_config(self, filepath):
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)

    # Update Agent Initialization
    @agent
    def problem_definition_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['problem_definition_agent']
        )

    @agent
    def problem_analyzer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['problem_analyzer_agent']
        )

    @agent
    def brainstorm_solutions_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['brainstorm_solutions_agent']
        )

    @agent
    def solution_evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['solution_evaluation_agent']
        )

    @agent
    def knowledge_graph_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['knowledge_graph_agent']
        )

    # Update Task Initialization
    @task
    def user_interview_task(self) -> Task:
        return Task(
            config=self.tasks_config['user_interview_task'],
            agent=self.problem_definition_agent()
        )

    @task
    def collect_data_task(self) -> Task:
        return Task(
            config=self.tasks_config['collect_data_task'],
            agent=self.problem_definition_agent()
        )

    @task
    def identify_gaps_task(self) -> Task:
        return Task(
            config=self.tasks_config['identify_gaps_task'],
            agent=self.problem_definition_agent()
        )

    @task
    def define_problem_task(self) -> Task:
        return Task(
            config=self.tasks_config['define_problem_task'],
            agent=self.problem_definition_agent()
        )

    @task
    def breakdown_problem_task(self) -> Task:
        return Task(
            config=self.tasks_config['breakdown_problem_task'],
            agent=self.problem_analyzer_agent()
        )

    @task
    def observe_patterns_task(self) -> Task:
        return Task(
            config=self.tasks_config['observe_patterns_task'],
            agent=self.problem_analyzer_agent()
        )

    @task
    def asses_resources_task(self) -> Task:
        return Task(
            config=self.tasks_config['asses_resources_task'],
            agent=self.problem_analyzer_agent()
        )

    @task
    def brainstorm_session_task(self) -> Task:
        return Task(
            config=self.tasks_config['brainstorm_session_task'],
            agent=self.brainstorm_solutions_agent()
        )

    @task
    def categorize_solutions_task(self) -> Task:
        return Task(
            config=self.tasks_config['categorize_solutions_task'],
            agent=self.brainstorm_solutions_agent()
        )

    @task
    def evaluate_solutions_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_solutions_task'],
            agent=self.solution_evaluation_agent()
        )

    @task
    def prioritize_solutions_task(self) -> Task:
        return Task(
            config=self.tasks_config['prioritize_solutions_task'],
            agent=self.solution_evaluation_agent()
        )

    @task
    def create_initial_graph(self) -> Task:
        return Task(
            config=self.tasks_config['create_initial_graph'],
            agent=self.knowledge_graph_agent()
        )

    @task
    def update_knowledge_graph(self) -> Task:
        return Task(
            config=self.tasks_config['update_knowledge_graph'],
            agent=self.knowledge_graph_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Problem Solving Crew"""
        return Crew(
            agents=[
                self.problem_definition_agent(),
                self.problem_analyzer_agent(),
                self.brainstorm_solutions_agent(),
                self.solution_evaluation_agent(),
                self.knowledge_graph_agent()
            ],
            tasks=[
                self.user_interview_task(),
                self.collect_data_task(),
                self.identify_gaps_task(),
                self.define_problem_task(),
                self.breakdown_problem_task(),
                self.observe_patterns_task(),
                self.asses_resources_task(),
                self.brainstorm_session_task(),
                self.categorize_solutions_task(),
                self.evaluate_solutions_task(),
                self.prioritize_solutions_task(),
                self.create_initial_graph(),
                self.update_knowledge_graph()
            ],
            process=Process.sequential,
            verbose=True
        )

    def kickoff(self, inputs):
        crew = self.crew()
        process = crew.create_process()

        # Handle human input for tasks that require it
        for task in process.tasks:
            if task.human_input:
                user_input = input(f"Please provide input for the '{task.name}' task: ")
                process.set_input(task.name, user_input)

        process.execute()
        return process.output