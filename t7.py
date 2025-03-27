import streamlit as st
import random


# Helper functions (same as before)
def is_valid_chromosome(chromosome, num_plants=7, num_seasons=4):
    """
    بررسی اعتبار کروموزوم بر اساس محدودیت‌های مسئله.
    نیروگاه‌های ۱ و ۲: دقیقاً یک دوره تعمیر دو فصل متوالی
    نیروگاه‌های ۳ تا ۷: دقیقاً یک فصل تعمیر
    """
    if all(bit == 0 for bit in chromosome):
        return False

    for plant in range(num_plants):
        start_index = plant * num_seasons
        end_index = start_index + num_seasons
        season_bits = chromosome[start_index:end_index]

        if plant < 2:  # نیروگاه‌های ۱ و ۲
            if sum(season_bits) != 2:
                return False
            # بررسی متوالی بودن دو فصل تعمیر
            consecutive = False
            for i in range(num_seasons):
                if season_bits[i] == 1 and season_bits[(i + 1) % num_seasons] == 1:
                    consecutive = True
                    break
            if not consecutive:
                return False
        else:  # نیروگاه‌های ۳ تا ۷
            if sum(season_bits) != 1:
                return False
    return True


def generate_valid_chromosome(num_plants=7, num_seasons=4):
    """
    تولید یک کروموزوم معتبر که تمام نیروگاه‌ها تعمیر شوند.
    """
    chromosome = [0] * (num_plants * num_seasons)

    # برای نیروگاه‌های ۱ و ۲: انتخاب دو فصل متوالی برای تعمیر
    for plant in range(2):
        start_season = random.randint(0, num_seasons - 1)
        chromosome[plant * num_seasons + start_season] = 1
        chromosome[plant * num_seasons + (start_season + 1) % num_seasons] = 1

    # برای نیروگاه‌های ۳ تا ۷: انتخاب یک فصل تصادفی برای تعمیر
    for plant in range(2, num_plants):
        season = random.randint(0, num_seasons - 1)
        chromosome[plant * num_seasons + season] = 1

    return chromosome


def generate_population(population_size=50, num_plants=7, num_seasons=4):
    """
    تولید جمعیت اولیه از کروموزوم‌های معتبر.
    """
    population = []
    for _ in range(population_size):
        chromosome = generate_valid_chromosome(num_plants, num_seasons)
        population.append(chromosome)
    return population


def decode_chromosome(chromosome, num_plants=7, num_seasons=4):
    """
    تبدیل کروموزوم به برنامه‌ی تعمیرات نیروگاه‌ها.
    """
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
    """
    محاسبه ذخیره خالص در هر بازه زمانی.
    """
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
    """
    محاسبه هزینه کل تعمیرات.
    """
    total_cost = 0
    for plant in range(num_plants):
        start_index = plant * num_seasons
        for season in range(num_seasons):
            if chromosome[start_index + season] == 1:
                total_cost += maintenance_costs[plant][season]
    return total_cost


def fitness_function(chromosome, demand, capacity, maintenance_costs, w1=10, w2=0.1, budget=None, penalty=10000):
    """
    تابع ارزیابی ترکیبی با استفاده از جمع وزنی و اعمال محدودیت‌ها.
    """
    if all(bit == 0 for bit in chromosome):
        return -penalty

    net_reserves = calculate_net_reserve(chromosome, demand, capacity)
    min_net_reserve = min(net_reserves)

    if min_net_reserve < 0:
        return -penalty

    total_maintenance_cost = calculate_maintenance_cost(chromosome, maintenance_costs)

    if budget is not None and total_maintenance_cost > budget:
        return -penalty

    fitness = w1 * min_net_reserve - w2 * total_maintenance_cost
    return fitness


def ensure_valid_population(population, demand, capacity, maintenance_costs, w1=10, w2=0.1, budget=None, penalty=10000):
    """
    اطمینان از معتبر بودن تمام کروموزوم‌ها در جمعیت.
    """
    while True:
        fitness_scores = [fitness_function(chromosome, demand, capacity, maintenance_costs, w1, w2, budget) for
                          chromosome in population]

        if -penalty not in fitness_scores:
            break

        for i in range(len(population)):
            if fitness_scores[i] == -penalty:
                population[i] = generate_valid_chromosome()

    return population


def roulette_wheel_selection(population, fitness_scores):
    """
    انتخاب والدین با استفاده از روش چرخ رولت.
    """
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    selected = []
    for _ in range(len(population)):
        selected_index = random.choices(range(len(population)), weights=probabilities)[0]
        selected.append(population[selected_index])
    return selected


def chunk_crossover(parent1, parent2, crossover_rate=0.8, chunk_size=4):
    """
    تقاطع بلوکی (Chunk Crossover) با بلوک‌های ۴ بیتی و احتمال کراس‌اور.
    """
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


def ensure_valid_child(child, demand, capacity, maintenance_costs, w1, w2, budget, penalty=10000):
    """
    اطمینان از معتبر بودن کروموزوم فرزند با رعایت تمام محدودیت‌ها
    """
    while True:
        # بررسی اعتبار ساختاری
        if not is_valid_chromosome(child):
            child = generate_valid_chromosome()
            continue

        # بررسی کفایت ذخیره انرژی
        net_reserves = calculate_net_reserve(child, demand, capacity)
        if min(net_reserves) < 0:
            child = generate_valid_chromosome()
            continue

        # بررسی بودجه
        total_maintenance_cost = calculate_maintenance_cost(child, maintenance_costs)
        if budget is not None and total_maintenance_cost > budget:
            child = generate_valid_chromosome()
            continue

        break
    return child


def mutate(chromosome, mutation_rate=0.01, num_plants=7, num_seasons=4):
    """
    جهش (Mutation) با احتمال مشخص، با حفظ ساختار بلوک‌های ۴ بیتی.
    """
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


def execute_genetic_algorithm(population_size=50, max_generations=100, mutation_rate=0.01,
                              w1=1.0000, w2=0.0100, budget=None):
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
    population = ensure_valid_population(population, demand, capacity, maintenance_costs, w1, w2, budget)

    current_generation = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    while current_generation < max_generations:
        progress = (current_generation + 1) / max_generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {current_generation + 1} of {max_generations}")

        initial_fitness = [fitness_function(chrom, demand, capacity, maintenance_costs, w1, w2, budget) for chrom in
                           population]

        selected_parents = roulette_wheel_selection(population, initial_fitness)

        new_population = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]
            child1, child2 = chunk_crossover(parent1, parent2, crossover_rate=0.8)

            child1 = ensure_valid_child(child1, demand, capacity, maintenance_costs, w1, w2, budget)
            child2 = ensure_valid_child(child2, demand, capacity, maintenance_costs, w1, w2, budget)

            new_population.append(child1)
            new_population.append(child2)

        for i in range(len(new_population)):
            new_population[i] = mutate(new_population[i], mutation_rate)

        for i in range(len(new_population)):
            if not is_valid_chromosome(new_population[i]):
                new_population[i] = generate_valid_chromosome()

            net_reserves = calculate_net_reserve(new_population[i], demand, capacity)
            if min(net_reserves) < 0:
                new_population[i] = generate_valid_chromosome()

        population = new_population
        current_generation += 1

    progress_bar.empty()
    status_text.empty()

    final_fitness = [fitness_function(chrom, demand, capacity, maintenance_costs, w1, w2, budget) for chrom in
                     population]
    best_chromosome = population[final_fitness.index(max(final_fitness))]
    best_fitness = max(final_fitness)

    with results_container:
        st.subheader("Best Solution Found")
        st.write("Chromosome:", best_chromosome)
        st.write("Fitness:", best_fitness)

        net_reserves = calculate_net_reserve(best_chromosome, demand, capacity)
        min_net_reserve = min(net_reserves)
        total_maintenance_cost = calculate_maintenance_cost(best_chromosome, maintenance_costs)

        st.write("Net Reserves:", net_reserves)
        st.write("Minimum Net Reserve:", min_net_reserve)
        st.write("Total Maintenance Cost:", total_maintenance_cost)

        st.subheader("Maintenance Schedule")
        best_schedule = decode_chromosome(best_chromosome)
        for plant, schedule in best_schedule.items():
            st.write(f"{plant}:")
            st.json(schedule)


def main():
    st.set_page_config(page_title="Power Plant Maintenance Scheduling", layout="wide")
    st.title("Power Plant Maintenance Scheduling with Genetic Algorithm")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Algorithm Parameters")
        population_size = st.number_input("Population Size", min_value=10, max_value=500, value=50)
        max_generations = st.number_input("Max Generations", min_value=10, max_value=1000, value=100)
        mutation_rate = st.number_input("Mutation Rate", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        w1 = st.number_input("w1 (Net Reserve Weight)", min_value=0.0, max_value=10.0, value=1.0, step=0.0001,
                             format="%.4f")
        w2 = st.number_input("w2 (Cost Weight)", min_value=0.0, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
        budget = st.number_input("Budget (optional)", min_value=0, value=1000)

        if st.button("Run Algorithm"):
            with st.spinner("Running genetic algorithm..."):
                execute_genetic_algorithm(
                    population_size=population_size,
                    max_generations=max_generations,
                    mutation_rate=mutation_rate,
                    w1=w1,
                    w2=w2,
                    budget=budget if budget > 0 else None
                )

    with col2:
        st.header("w1 and w2 Interpretation")
        st.markdown("""
        **جدول راهنما**

        | تاثیر | w1    | w2    | تست    |
        |---|---|---|---|
        | تأکید بسیار زیاد بر ذخیره خالص (ممكن است هزینه‌ها بسیار بالا بروند) | 10    | 0.1   | 1    |
        | توزن بهتر (بیشنهاد اصلی) | 1    | 0.01   | 2    |
        | تأکید کمی بیشتر بر ذخیره خالص | 1    | 0.005   | 3    |
        | تأکید بیشتر بر کاهش هزینه | 0.5    | 0.001   | 4    |

         **نتیجه گیری**  

        به اولویت‌های مسئله بستگی دارد w2 و w1 انتخاب مقادیر مناسب  

        اگر تأمین برق بسیار حیاتی است: تست 1 یا 3  

        اگر توان بین تأمین برق و هزینه مهم است: تست 2 (پیشنهاد اصلی)  

        اگر کنترل هزینه‌ها اهمیت بیشتری دارد: تست 4  

        برای انتخاب بهترین تنظیم، می‌توانید نتایج هر چهار تست را مقایسه کنید و بیبینید.  
        کدام یک بهترین تعادل بین ذخیره خالص و هزینه تعمیرات را ایجاد می‌کند.
        """)


if __name__ == "__main__":
    main()