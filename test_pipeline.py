import sys
sys.path.insert(0, "./src")
from src.pipelines.query import QueryPipeline

print("Instantiating pipeline...")
pipe = QueryPipeline()
print("Running query...")
res = pipe.run("What is regression?")
print("Result:", res)
