from src.models.router_llm import RouterLLM
from src.agents.query_router import QueryRouter

llm = RouterLLM()

router = QueryRouter(llm)

queries = [

    "hello",

    "what is quantum tunneling",

    "tell me about it",

    "write bfs in c++",

    "2+2"
]

for q in queries:

    print(q)

    print(router.route(q))

    print("-"*50)