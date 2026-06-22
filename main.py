from dotenv import load_dotenv

from agents.demo import run_demo


def main():
    load_dotenv()
    run_demo()


if __name__ == "__main__":
    main()
