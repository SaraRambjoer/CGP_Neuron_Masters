stops = [1, 2, 3, 5, 10, 20, 50]
fitness = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.25, 0.50, 0.75]
runs = 100

for stop in stops:
    for fit in fitness:
        print(fit*(runs-stop), end=", ")
    print()
