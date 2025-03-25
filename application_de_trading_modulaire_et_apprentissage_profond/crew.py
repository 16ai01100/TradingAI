from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import GithubSearchTool
from crewai_tools import CodeDocsSearchTool
from crewai_tools import JSONSearchTool
from crewai_tools import DirectoryReadTool
from crewai_tools import DirectorySearchTool

@CrewBase
class ApplicationDeTradingModulaireEtApprentissageProfondCrew():
    """ApplicationDeTradingModulaireEtApprentissageProfond crew"""

    @agent
    def data_collector(self) -> Agent:
        return Agent(
            config=self.agents_config['data_collector'],
            tools=[ScrapeWebsiteTool()],
        )

    @agent
    def strategy_executor(self) -> Agent:
        return Agent(
            config=self.agents_config['strategy_executor'],
            tools=[GithubSearchTool(), CodeDocsSearchTool()],
        )

    @agent
    def ml_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['ml_developer'],
            tools=[GithubSearchTool()],
        )

    @agent
    def backtesting_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backtesting_engineer'],
            tools=[JSONSearchTool()],
        )

    @agent
    def web_interface_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['web_interface_designer'],
            tools=[DirectoryReadTool(), DirectorySearchTool()],
        )


    @task
    def configure_data_environment(self) -> Task:
        return Task(
            config=self.tasks_config['configure_data_environment'],
            tools=[ScrapeWebsiteTool()],
        )

    @task
    def develop_trading_strategies(self) -> Task:
        return Task(
            config=self.tasks_config['develop_trading_strategies'],
            tools=[GithubSearchTool(), CodeDocsSearchTool()],
        )

    @task
    def build_ml_module(self) -> Task:
        return Task(
            config=self.tasks_config['build_ml_module'],
            tools=[GithubSearchTool()],
        )

    @task
    def perform_backtesting(self) -> Task:
        return Task(
            config=self.tasks_config['perform_backtesting'],
            tools=[JSONSearchTool()],
        )

    @task
    def design_web_interface(self) -> Task:
        return Task(
            config=self.tasks_config['design_web_interface'],
            tools=[DirectoryReadTool(), DirectorySearchTool()],
        )

    @task
    def integration_testing(self) -> Task:
        return Task(
            config=self.tasks_config['integration_testing'],
            tools=[GithubSearchTool()],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the ApplicationDeTradingModulaireEtApprentissageProfond crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
