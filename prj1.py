import streamlit as st
import random
from fpdf import FPDF
import base64


# Helper functions (same as before)
def is_valid_chromosome(chromosome, num_plants=7, num_seasons=4):
    """بررسی اعتبار کروموزوم با دقت بیشتر"""
    if all(bit == 0 for bit in chromosome):
        return False

    for plant in range(num_plants):
        start = plant * num_seasons
        seasons = chromosome[start:start + num_seasons]

        if plant < 2:  # نیروگاه‌های 1 و 2
            if sum(seasons) != 2:
                return False
            # بررسی متوالی بودن
            consecutive = any(seasons[i] == 1 and seasons[(i + 1) % num_seasons] == 1
                              for i in range(num_seasons))
            if not consecutive:
                return False
        else:  # نیروگاه‌های 3 تا 7
            if sum(seasons) != 1:
                return False

    return True


def generate_valid_chromosome(num_plants=7, num_seasons=4):
    """تولید کروموزوم معتبر با اطمینان از تعمیر تمام نیروگاه‌ها"""
    chromosome = [0] * (num_plants * num_seasons)

    # نیروگاه‌های 1 و 2: دو فصل متوالی تعمیر
    for plant in range(2):
        start_season = random.randint(0, num_seasons - 1)
        chromosome[plant * num_seasons + start_season] = 1
        chromosome[plant * num_seasons + (start_season + 1) % num_seasons] = 1

    # نیروگاه‌های 3 تا 7: یک فصل تعمیر
    for plant in range(2, num_plants):
        season = random.randint(0, num_seasons - 1)
        chromosome[plant * num_seasons + season] = 1

    # اعتبارسنجی نهایی
    if not is_valid_chromosome(chromosome):
        return generate_valid_chromosome(num_plants, num_seasons)
    return chromosome


def fitness_function(chromosome, demand, capacity, maintenance_costs, w1=10, w2=0.1, budget=None, penalty=10000):
    """تابع ارزیابی با جریمه سنگین برای حالت‌های نامعتبر"""
    if not is_valid_chromosome(chromosome):
        return -penalty

    net_reserves = calculate_net_reserve(chromosome, demand, capacity)
    if min(net_reserves) < 0:
        return -penalty

    total_maintenance_cost = calculate_maintenance_cost(chromosome, maintenance_costs)
    if budget is not None and total_maintenance_cost > budget:
        return -penalty

    # محاسبه تعداد نیروگاه‌هایی که تعمیر شده‌اند
    num_plants = len(capacity)
    num_seasons = len(demand)
    repaired_plants = 0
    for plant in range(num_plants):
        start = plant * num_seasons
        if sum(chromosome[start:start + num_seasons]) > 0:
            repaired_plants += 1

    # جریمه اگر همه نیروگاه‌ها تعمیر نشده‌باشند
    if repaired_plants < num_plants:
        return -penalty

    return w1 * min(net_reserves) - w2 * total_maintenance_cost


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


def create_pdf_report(best_chromosome, best_fitness, net_reserves, min_net_reserve,
                      total_maintenance_cost, best_schedule, best_fitness_per_generation,
                      demand, capacity, maintenance_costs):
    """تابع برای ایجاد گزارش PDF از نتایج"""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # عنوان گزارش
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Power Plant Maintenance Scheduling Report", ln=1, align='C')
    pdf.ln(10)

    # بخش هزینه‌های تعمیرات
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Maintenance Costs per Plant and Season", ln=1)
    pdf.set_font("Arial", size=10)

    # ایجاد جدول هزینه‌ها
    col_width = 30
    row_height = 5
    seasons = ["Season 1", "Season 2", "Season 3", "Season 4"]

    # سرستون‌ها
    pdf.cell(col_width, row_height, txt="Plant", border=1, align='C')
    for season in seasons:
        pdf.cell(col_width, row_height, txt=season, border=1, align='C')
    pdf.ln(row_height)

    # داده‌های هزینه‌ها
    for plant_idx in range(len(maintenance_costs)):
        pdf.cell(col_width, row_height, txt=f"Plant {plant_idx + 1}", border=1, align='C')
        for cost in maintenance_costs[plant_idx]:
            pdf.cell(col_width, row_height, txt=str(cost), border=1, align='C')
        pdf.ln(row_height)

    pdf.ln(10)

    # اطلاعات کلی
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Optimal Solution Summary", ln=1)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Best Fitness: {best_fitness:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Minimum Net Reserve: {min_net_reserve}", ln=1)
    pdf.cell(200, 10, txt=f"Total Maintenance Cost: {total_maintenance_cost}", ln=1)
    pdf.ln(5)

    # اطلاعات فنی
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Technical Details", ln=1)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Chromosome: {best_chromosome}", ln=1)
    pdf.ln(5)

    pdf.cell(200, 10, txt="Net Reserves per Season:", ln=1)
    for i, reserve in enumerate(net_reserves, 1):
        pdf.cell(200, 10, txt=f"  Season {i}: {reserve}", ln=1)
    pdf.ln(5)

    # برنامه تعمیرات
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Maintenance Schedule", ln=1)
    pdf.set_font("Arial", size=12)

    for plant, schedule in best_schedule.items():
        pdf.cell(200, 10, txt=f"{plant}:", ln=1)
        for season, status in schedule.items():
            pdf.cell(200, 10, txt=f"  {season}: {status}", ln=1)
        pdf.ln(2)

    # روند بهبود فیتنس
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Fitness Improvement Over Generations", ln=1)
    pdf.set_font("Arial", size=12)

    for gen, fitness in enumerate(best_fitness_per_generation, 1):
        pdf.cell(200, 10, txt=f"Generation {gen}: {fitness:.2f}", ln=1)

    # اطلاعات ورودی
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Input Parameters", ln=1)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Demand per Season:", ln=1)
    for i, d in enumerate(demand, 1):
        pdf.cell(200, 10, txt=f"  Season {i}: {d}", ln=1)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Plant Capacities:", ln=1)
    for i, cap in enumerate(capacity, 1):
        pdf.cell(200, 10, txt=f"  Plant {i}: {cap}", ln=1)

    return pdf


def get_pdf_download_link(pdf, filename):
    """تابع برای ایجاد لینک دانلود PDF"""
    pdf_output = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href


def execute_genetic_algorithm(population_size=50, max_generations=100, mutation_rate=0.01,
                              w1=1.0000, w2=0.0100, budget=None):
    # پارامترهای مسئله
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

    # ایجاد جمعیت اولیه
    population = generate_population(population_size)
    population = ensure_valid_population(population, demand, capacity, maintenance_costs, w1, w2, budget)

    # متغیرهای ردیابی بهترین جواب
    best_overall_chromosome = None
    best_overall_fitness = -float('inf')
    best_fitness_per_generation = []
    current_generation = 0

    # ایجاد عناصر رابط کاربری
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    # حلقه اصلی الگوریتم ژنتیک
    while current_generation < max_generations:
        # به روزرسانی رابط کاربری
        progress = (current_generation + 1) / max_generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {current_generation + 1} of {max_generations}")

        # محاسبه فیتنس برای هر کروموزوم
        fitness_scores = [fitness_function(chrom, demand, capacity, maintenance_costs, w1, w2, budget)
                          for chrom in population]

        # یافتن بهترین کروموزوم در این نسل
        current_best_fitness = max(fitness_scores)
        current_best_index = fitness_scores.index(current_best_fitness)
        current_best_chromosome = population[current_best_index]

        # به روزرسانی بهترین جواب کلی
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_chromosome = current_best_chromosome.copy()

        best_fitness_per_generation.append(current_best_fitness)

        # انتخاب والدین
        selected_parents = roulette_wheel_selection(population, fitness_scores)

        # ایجاد نسل جدید
        new_population = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]

            # انجام تقاطع
            child1, child2 = chunk_crossover(parent1, parent2, crossover_rate=0.8)

            # اعتبارسنجی فرزندان
            child1 = ensure_valid_child(child1, demand, capacity, maintenance_costs, w1, w2, budget)
            child2 = ensure_valid_child(child2, demand, capacity, maintenance_costs, w1, w2, budget)

            new_population.append(child1)
            new_population.append(child2)

        # اعمال جهش
        for i in range(len(new_population)):
            new_population[i] = mutate(new_population[i], mutation_rate)

        # جایگزینی جمعیت قدیم با جدید
        population = new_population
        current_generation += 1

    # پاکسازی عناصر رابط کاربری
    progress_bar.empty()
    status_text.empty()

    # محاسبه نتایج نهایی بر اساس بهترین کروموزوم کلی
    net_reserves = calculate_net_reserve(best_overall_chromosome, demand, capacity)
    min_net_reserve = min(net_reserves)
    total_maintenance_cost = calculate_maintenance_cost(best_overall_chromosome, maintenance_costs)
    best_schedule = decode_chromosome(best_overall_chromosome)

    # نمایش نتایج
    with results_container:
        st.subheader("Best Overall Solution Found")
        st.write("Chromosome:", best_overall_chromosome)
        st.write("Fitness:", best_overall_fitness)
        st.write("Net Reserves:", net_reserves)
        st.write("Minimum Net Reserve:", min_net_reserve)
        st.write("Total Maintenance Cost:", total_maintenance_cost)

        st.subheader("Maintenance Schedule")
        for plant, schedule in best_schedule.items():
            st.write(f"{plant}:")
            st.json(schedule)

        st.subheader("Best Fitness per Generation")
        for gen, fitness in enumerate(best_fitness_per_generation, 1):
            st.text(f"Generation {gen} -> Best Fitness = {fitness:.2f}")

        # ایجاد و نمایش لینک دانلود PDF
        pdf_report = create_pdf_report(
            best_overall_chromosome, best_overall_fitness, net_reserves, min_net_reserve,
            total_maintenance_cost, best_schedule, best_fitness_per_generation,
            demand, capacity, maintenance_costs
        )

        st.markdown(get_pdf_download_link(pdf_report, "Maintenance_Schedule_Report.pdf"), unsafe_allow_html=True)


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