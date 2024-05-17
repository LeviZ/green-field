from dotenv import load_dotenv
load_dotenv()

from crew import ProblemSolvingCrew

def run():
    inputs = {
        'user_problem': input("Enter your high level problem/goal: ")
    }
    crew = ProblemSolvingCrew()
    crew.kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
