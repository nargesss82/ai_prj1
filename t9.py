import streamlit as st
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import matplotlib.pyplot as plt


# Helper functions
def is_valid_chromosome(chromosome, num_plants=7, num_seasons=4):
    if all(bit == 0 for bit in chromosome):
        return False

    for plant in range(num_plants):
        start_index = plant * num_seasons
        end_index = start_index + num_seasons
        season_bits = chromosome[start_index:end_index]

        if plant < 2:  # نیروگاه‌های ۱ و ۲
            if sum(season_bits) == 0:
                continue
            if sum(season_bits) != 2:
                return False
            consecutive_pairs = [
                (season_bits[0], season_bits[1]),
                (season_bits[1], season_bits[2]),
                (season_bits[2], season_bits[3]),
                (season_bits[3], season_bits[0])
            ]
            if not any(pair == (1, 1) for pair in consecutive_pairs):
                return False
        else:  # نیروگاه‌های ۳ تا ۷
            if sum(season_bits) > 1:
                return False
    return True


def generate_valid_chromosome(num_plants=7, num_seasons=4):
    while True:
        chromosome = [random.randint(0, 1) for _ in range(num_plants * num_seasons)]
        if is_valid_chromosome(chromosome):
            return chromosome


def generate_population(population_size=50, num_plants=7, num_seasons=4):
    population = []
    for _ in range(population_size):
        chromosome = generate_valid_chromosome(num_plants, num_seasons)
        population.append(chromosome)
    return population


def decode_chromosome(chromosome, num_plants=7, num_seasons=4):
    schedule = {}
    for plant in range(num_plants):
        start_index = plant * num_seasons
        end_index = start_index + num_seasons
        season_bits = chromosome[start_index:end_index]
        schedule[f"Plant {plant + 1}"] = {
            f"Season {season + 1}": "Under Maintenance" if bit == 1 else "Operational"
            for season, bit in enumerate(season_bits)
        }
    return schedule


def calculate_net_reserve(chromosome, demand, capacity, num_plants=7, num_seasons=4):
    net_reserves = []
    for season in range(num_seasons):
        total_capacity = 0
        for plant in range(num_plants):
            start_index = plant * num_seasons
            if chromosome[start_index + season] == 0:
                total_capacity += capacity[plant]
        net_reserve = total_capacity - demand[season]
        net_reserves.append(net_reserve)
    return net_reserves


def calculate_maintenance_cost(chromosome, maintenance_costs, num_plants=7, num_seasons=4):
    total_cost = 0
    for plant in range(num_plants):
        start_index = plant * num_seasons
        for season in range(num_seasons):
            if chromosome[start_index + season] == 1:
                total_cost += maintenance_costs[plant][season]
    return total_cost


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fast_non_dominated_sort(population, demand, capacity, maintenance_costs):
    fronts = [[]]
    domination_counts = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]
    ranks = [0] * len(population)

    fitness_values = []
    for chrom in population:
        net_reserves = calculate_net_reserve(chrom, demand, capacity)
        min_net_reserve = min(net_reserves)
        total_cost = calculate_maintenance_cost(chrom, maintenance_costs)
        fitness_values.append((min_net_reserve, -total_cost))

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if dominates(fitness_values[i], fitness_values[j]):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates(fitness_values[j], fitness_values[i]):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    ranks[j] = current_front + 1
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return fronts, ranks


def crowding_distance_assignment(front, population, demand, capacity, maintenance_costs):
    distances = [0] * len(front)
    num_objectives = 2

    for m in range(num_objectives):
        objective_values = []
        for i in front:
            chrom = population[i]
            net_reserves = calculate_net_reserve(chrom, demand, capacity)
            min_net_reserve = min(net_reserves)
            total_cost = calculate_maintenance_cost(chrom, maintenance_costs)
            if m == 0:
                objective_values.append((i, min_net_reserve))
            else:
                objective_values.append((i, -total_cost))

        objective_values.sort(key=lambda x: x[1])

        distances[objective_values[0][0]] = float('inf')
        distances[objective_values[-1][0]] = float('inf')

        min_val = objective_values[0][1]
        max_val = objective_values[-1][1]
        if max_val == min_val:
            continue

        for i in range(1, len(objective_values) - 1):
            idx = objective_values[i][0]
            next_val = objective_values[i + 1][1]
            prev_val = objective_values[i - 1][1]
            distances[idx] += (next_val - prev_val) / (max_val - min_val)

    return distances


def nsga2_selection(population, demand, capacity, maintenance_costs, population_size):
    fronts, ranks = fast_non_dominated_sort(population, demand, capacity, maintenance_costs)

    new_population = []
    current_front = 0
    while len(new_population) + len(fronts[current_front]) <= population_size:
        new_population.extend([population[i] for i in fronts[current_front]])
        current_front += 1

    if len(new_population) < population_size:
        remaining = population_size - len(new_population)
        last_front = fronts[current_front]

        distances = crowding_distance_assignment(last_front, population, demand, capacity, maintenance_costs)

        sorted_last_front = sorted(zip(last_front, distances), key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in sorted_last_front[:remaining]]
        new_population.extend([population[i] for i in selected])

    return new_population


def chunk_crossover(parent1, parent2, crossover_rate=0.8, chunk_size=4):
    if random.random() < crossover_rate:
        num_chunks = len(parent1) // chunk_size
        crossover_point = random.randint(1, num_chunks - 1)
        start_index = crossover_point * chunk_size
        end_index = len(parent1)
        child1 = parent1[:start_index] + parent2[start_index:end_index]
        child2 = parent2[:start_index] + parent1[start_index:end_index]
    else:
        child1, child2 = parent1[:], parent2[:]
    return child1, child2


def mutate(chromosome, mutation_rate=0.01, num_plants=7, num_seasons=4):
    original_chromosome = chromosome.copy()

    for plant in range(num_plants):
        start_index = plant * num_seasons
        end_index = start_index + num_seasons
        if random.random() < mutation_rate:
            season_to_mutate = random.randint(0, num_seasons - 1)
            chromosome[start_index + season_to_mutate] = 1 - chromosome[start_index + season_to_mutate]

    if all(bit == 0 for bit in chromosome):
        return original_chromosome

    return chromosome


def ensure_valid_child(child, demand, capacity, maintenance_costs, budget=None, penalty=10000):
    while True:
        if not is_valid_chromosome(child):
            child = generate_valid_chromosome()
            continue

        net_reserves = calculate_net_reserve(child, demand, capacity)
        if min(net_reserves) < 0:
            child = generate_valid_chromosome()
            continue

        total_maintenance_cost = calculate_maintenance_cost(child, maintenance_costs)
        if budget is not None and total_maintenance_cost > budget:
            child = generate_valid_chromosome()
            continue

        break
    return child


def create_generation_report(generation_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Generation Report - Power Plant Maintenance Scheduling", styles['Title']))
    elements.append(Spacer(1, 0.25 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def execute_genetic_algorithm(population_size=50, max_generations=100, mutation_rate=0.01, budget=None):
    demand = [80, 90, 65, 70]
    capacity = [20, 15, 35, 40, 15, 15, 10]
    maintenance_costs = [
        [100, 120, 110, 130],
        [90, 95, 100, 105],
        [80, 85, 90, 95],
        [150, 160, 155, 165],
        [70, 75, 80, 85],
        [60, 65, 70, 75],
        [50, 55, 60, 65]
    ]

    population = generate_population(population_size)
    current_generation = 0
    generation_data = []

    population = [chrom for chrom in population if is_valid_chromosome(chrom)]
    while len(population) < population_size:
        population.append(generate_valid_chromosome())

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    while current_generation < max_generations:
        progress = (current_generation + 1) / max_generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {current_generation + 1} of {max_generations}")

        parents = nsga2_selection(population, demand, capacity, maintenance_costs, population_size)

        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = chunk_crossover(parent1, parent2)

            child1 = ensure_valid_child(child1, demand, capacity, maintenance_costs, budget)
            child2 = ensure_valid_child(child2, demand, capacity, maintenance_costs, budget)

            offspring.append(child1)
            offspring.append(child2)

        for i in range(len(offspring)):
            offspring[i] = mutate(offspring[i], mutation_rate)

        combined_population = population + offspring
        population = nsga2_selection(combined_population, demand, capacity, maintenance_costs, population_size)

        current_generation += 1

    progress_bar.empty()
    status_text.empty()

    fronts, _ = fast_non_dominated_sort(population, demand, capacity, maintenance_costs)
    pareto_front = [population[i] for i in fronts[0]]

    with results_container:
        st.subheader("Pareto Optimal Solutions")

        pareto_fitness = []
        for chrom in pareto_front:
            net_reserves = calculate_net_reserve(chrom, demand, capacity)
            min_net_reserve = min(net_reserves)
            total_cost = calculate_maintenance_cost(chrom, maintenance_costs)
            pareto_fitness.append((min_net_reserve, total_cost))

        st.write("Number of Pareto Solutions:", len(pareto_front))

        fig, ax = plt.subplots()
        costs = [x[1] for x in pareto_fitness]
        reserves = [x[0] for x in pareto_fitness]
        ax.scatter(costs, reserves)
        ax.set_xlabel('Total Maintenance Cost')
        ax.set_ylabel('Minimum Net Reserve')
        ax.set_title('Pareto Front')
        st.pyplot(fig)

        for i, (chrom, (min_reserve, cost)) in enumerate(zip(pareto_front, pareto_fitness)):
            st.write(f"\nSolution {i + 1}:")
            st.write(f"- Min Net Reserve: {min_reserve}")
            st.write(f"- Total Maintenance Cost: {cost}")

            st.write("Maintenance Schedule:")
            schedule = decode_chromosome(chrom)
            for plant, plant_schedule in schedule.items():
                st.write(f"{plant}: {plant_schedule}")

    return generation_data


def main():
    st.set_page_config(page_title="Power Plant Maintenance Scheduling", layout="wide")
    st.title("Power Plant Maintenance Scheduling with NSGA-II")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Algorithm Parameters")
        population_size = st.number_input("Population Size", min_value=10, max_value=500, value=50)
        max_generations = st.number_input("Max Generations", min_value=10, max_value=1000, value=100)
        mutation_rate = st.number_input("Mutation Rate", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        budget = st.number_input("Budget (optional)", min_value=0, value=1000)

        if st.button("Run NSGA-II Algorithm"):
            with st.spinner("Running NSGA-II algorithm..."):
                generation_data = execute_genetic_algorithm(
                    population_size=population_size,
                    max_generations=max_generations,
                    mutation_rate=mutation_rate,
                    budget=budget if budget > 0 else None
                )

                pdf_buffer = create_generation_report(generation_data)
                st.download_button(
                    label="Download Generation Report",
                    data=pdf_buffer,
                    file_name="nsga2_generation_report.pdf",
                    mime="application/pdf"
                )

    with col2:
        st.header("NSGA-II Explanation")
        st.markdown("""
        **الگوریتم NSGA-II (Non-dominated Sorting Genetic Algorithm II)**

        - یک الگوریتم چندهدفه معروف
        - جبهه پارتو را شناسایی می‌کند
        - از دو مکانیزم اصلی استفاده می‌کند:
          1. مرتب‌سازی غیرمسلط
          2. فاصله ازدحام

        **مزایا:**
        - پیدا کردن مجموعه‌ای از راه‌حل‌های بهینه
        - حفظ تنوع در جمعیت
        - مناسب برای مسائل بهینه‌سازی چندهدفه

        **خروجی:**
        - مجموعه‌ای از راه‌حل‌های غیرمسلط (جبهه پارتو)
        - می‌توانید بر اساس نیاز خود یکی را انتخاب کنید
        """)


if __name__ == "__main__":
    main()