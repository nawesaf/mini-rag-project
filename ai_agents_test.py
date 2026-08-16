from time import perf_counter

from ai_agent_from_scratch import ReAct_agent_loop, planner_agent_loop

TEST_QUERIES = [
    "Give me the files available in the project.",
    "Find where the ReAct agent loop is defined and explain its role.",
    "Explain how tools are executed in the agent.",
]

FILE_CREATION_QUERY = """Create a new file named `integer_addition.py` in the project root.
In that file, implement a typed Python function named `add_integers` that accepts two
integers and returns their sum. Include a concise docstring. Do not modify any other file."""


def test_agent(name, agent, query):
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"Query: {query}")
    print("=" * 60)

    start = perf_counter()

    try:
        result = agent(query, debug=True)
        elapsed = perf_counter() - start

        print(f"\nResult:\n{result}")
        print(f"\nTime: {elapsed:.2f}s")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")


def main():
    # for query in TEST_QUERIES:
    #     print(f"\n\n{'#' * 70}")
    #     print(f"TEST: {query}")
    #     print("#" * 70)

    #     test_agent(
    #         "Simple ReAct agent",
    #         ReAct_agent_loop,
    #         query
    #     )

    #     test_agent(
    #         "Planner + ReAct agent",
    #         planner_agent_loop,
    #         query
    #     )

    # print(f"\n\n{'#' * 70}")
    print("TEST: Create and implement a new Python file")
    print("#" * 70)

    test_agent(
        "Planner + ReAct agent — file creation",
        planner_agent_loop,
        FILE_CREATION_QUERY,
    )


if __name__ == "__main__":
    main()
