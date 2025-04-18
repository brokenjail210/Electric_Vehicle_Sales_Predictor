import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def add_footer(fig):
    footer = "© 2025 Aditya Arora | www.linkedin.com/in/NeuralAditya"
    fig.text(0.5, 0.01, footer, ha='center', fontsize=8, color='gray')

def generate_graphs(
    df_original,
    selected_class: str = None,
    selected_category: str = None,
    selected_type: str = None,
    output_dir: str = "static/graphs"
):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "graphs_report.pdf")
    pdf = PdfPages(pdf_path)

    # ——————————————————————————————
    # 1. EV Sales Trend Over Time
    # ——————————————————————————————
    if 'Date' in df_original.columns and 'EV_Sales_Quantity' in df_original.columns:
        df = df_original.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year']  = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        df_time = (
            df
            .groupby(['Year','Month'])['EV_Sales_Quantity']
            .sum()
            .reset_index()
        )
        df_time['Date'] = pd.to_datetime(df_time[['Year','Month']].assign(DAY=1))
        df_time.sort_values('Date', inplace=True)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_time, x='Date', y='EV_Sales_Quantity', marker='o', ax=ax1)
        ax1.set_title("EV Sales Trend Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("EV Sales Quantity")
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
        ax1.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        add_footer(fig1)
        fig1.savefig(os.path.join(output_dir, "ev_sales_trend.png"))
        pdf.savefig(fig1)
        plt.close(fig1)

    # ——————————————————————————————
    # 2. EV Sales by State (filtered)
    # ——————————————————————————————
    needed_cols = {'State','Vehicle_Class','Vehicle_Category','Vehicle_Type','EV_Sales_Quantity'}
    if needed_cols.issubset(df_original.columns):
        df = df_original.copy()

        # apply user selections if provided
        if selected_class:
            df = df[df['Vehicle_Class'] == selected_class]
        if selected_category:
            df = df[df['Vehicle_Category'] == selected_category]
        if selected_type:
            df = df[df['Vehicle_Type'] == selected_type]

        # group by state to get total sales of that exact combo
        state_sales = (
            df
            .groupby('State')['EV_Sales_Quantity']
            .sum()
            .sort_values(ascending=False)
        )

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        state_sales.plot(kind='bar', color='skyblue', ax=ax2)
        title_parts = [
            selected_class or "All Classes",
            selected_category or "All Categories",
            selected_type or "All Types"
        ]
        ax2.set_title("EV Sales by State: " + " / ".join(title_parts))
        ax2.set_xlabel("State")
        ax2.set_ylabel("Total EV Sales")
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        add_footer(fig2)
        fig2.savefig(os.path.join(output_dir, "ev_sales_by_state_filtered.png"))
        pdf.savefig(fig2)
        plt.close(fig2)

    # ——————————————————————————————
    # 3. Vehicle Type Distribution
    # ——————————————————————————————
    if 'Vehicle_Type' in df_original.columns:
        vehicle_distribution = (
            df_original
            .groupby('Vehicle_Type')['EV_Sales_Quantity']
            .sum()
            .sort_values(ascending=False)
        )
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.pie(
            vehicle_distribution,
            labels=vehicle_distribution.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette("pastel")
        )
        ax3.set_title("EV Sales Distribution by Vehicle Type")
        plt.tight_layout()
        add_footer(fig3)
        fig3.savefig(os.path.join(output_dir, "vehicle_type_distribution.png"))
        pdf.savefig(fig3)
        plt.close(fig3)

    # ——————————————————————————————
    # 4. Heatmap: Monthly EV Sales by State
    # ——————————————————————————————
    if {'State','Date','EV_Sales_Quantity'}.issubset(df_original.columns):
        df = df_original.copy()
        df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.month
        heatmap_data = (
            df
            .pivot_table(
                values='EV_Sales_Quantity',
                index='State',
                columns='Month',
                aggfunc='sum'
            )
            .fillna(0)
        )

        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            linewidths=.5,
            annot=True,
            fmt=".0f",
            ax=ax4
        )
        ax4.set_title("Monthly EV Sales by State")
        ax4.set_xlabel("Month")
        ax4.set_ylabel("State")
        plt.tight_layout()
        add_footer(fig4)
        fig4.savefig(os.path.join(output_dir, "state_month_heatmap.png"))
        pdf.savefig(fig4)
        plt.close(fig4)

    pdf.close()
    print(f"✅ Graphs saved to {output_dir} and PDF report generated.")

__all__ = ['generate_graphs']
