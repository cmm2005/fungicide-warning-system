import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings


def main():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    
    # Initialize the main window
    root = tk.Tk()
    root.title("Early Warning System for Fungicides")
    root.geometry("900x450")
    root.resizable(True, True)

    # Configure styles for labels - bold and size 12
    style = ttk.Style()
    style.configure("Bold.TLabel", font=("Arial", 12, "bold"))
    style.configure("Accent.TButton", padding=10)

    # Create container frames with borders for left and right sections
    left_container = ttk.Frame(root, padding=(10, 10), borderwidth=2, relief="solid")
    left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
    
    right_container = ttk.Frame(root, padding=(10, 10), borderwidth=2, relief="solid")
    right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

    # Left frame: Data input area inside the bordered container
    left_frame = ttk.Frame(left_container, padding=(10, 10))
    left_frame.pack(fill=tk.BOTH, expand=True)

    # Right frame: Result display area inside the bordered container
    right_frame = ttk.Frame(right_container, padding=(10, 10))
    right_frame.pack(fill=tk.BOTH, expand=True)

    # Draw a vertical separator between the containers
    separator = ttk.Separator(root, orient=tk.VERTICAL)
    separator.pack(side=tk.LEFT, fill=tk.Y, pady=20, padx=5)

    # Left frame: Data input section title (now inside the bordered container)
    ttk.Label(left_frame, text="Data Input", style="Bold.TLabel").grid(
        row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 20)
    )

    # Define options for Aquatic and Soil mediums
    aquatic_compounds = [
        "Compounds_Acetochlor", "Compounds_Azoxystrobin","Compounds_24-epibrassinolide", 
        "Compounds_Bromuconazole", "Compounds_Captan", "Compounds_Carbendazim",
        "Compounds_Chlorpyrifos", "Compounds_Difenoconazole", "Compounds_Econazole",
        "Compounds_Epoxiconazole", "Compounds_Fludioxonil", "Compounds_Fluopyram",
        "Compounds_Fluoxastrobin", "Compounds_Fluxapyroxad", "Compounds_Imidacloprid",
        "Compounds_Isopyrazam", "Compounds_Mancozeb", "Compounds_Metiram",
        "Compounds_Myclobutanil", "Compounds_Oxathiapiprolin", "Compounds_Propiconazole",
        "Compounds_Prothioconazole", "Compounds_Pyraclostrobin", "Compounds_R-(-)-penthiopyrad",
        "Compounds_R-metalaxyl", "Compounds_R-prothioconazole", "Compounds_R-prothioconazole-desthio",
        "Compounds_Rac-penthiopyrad", "Compounds_Rac-prothioconazole", "Compounds_Rac-prothioconazole-desthio",
        "Compounds_S-(+)-penthiopyrad", "Compounds_S-prothioconazole", "Compounds_S-prothioconazole-desthio",
        "Compounds_Salicylic acid", "Compounds_Salicylic acid with Thiram", "Compounds_Tebuconazole",
        "Compounds_Thifluzamide", "Compounds_Thiophanate-methyl", "Compounds_Thiram",
        "Compounds_Triadimefon", "Compounds_Trifloxystrobin", "Compounds_Triflumizole",
        "Compounds_Trimethyltin chloride"
    ]

    soil_compounds = [
        "Compounds_Azoxystrobin", "Compounds_Benzovindiflupyr", "Compounds_Carbendazim",
        "Compounds_Epoxiconazole", "Compounds_Flumorph", "Compounds_Fluopicolide",
        "Compounds_Fluoxastrobin", "Compounds_Hexaconazole", "Compounds_Hymexazol",
        "Compounds_Kitazin", "Compounds_Maneb", "Compounds_Mefentrifluconazole",
        "Compounds_Mefentrifluconazole ", "Compounds_Metalaxyl-M", "Compounds_Pentachloronitrobenzene",
        "Compounds_Pyraclostrobin", "Compounds_R-(-)-ptz", "Compounds_S-(+)-ptz",
        "Compounds_Tebuconazole", "Compounds_Thifluzamide", "Compounds_Thiram",
        "Compounds_Tolclofos-methyl", "Compounds_Triadimenol", "Compounds_Trifloxystrobin",
        "Compounds_Trifloxystrobin acid", "Compounds_Vinclozolin"
    ]

    aquatic_species = ["","Species_Zebrafish",
        "Species_Anabaena laxa", "Species_Caenorhabditis elegans", "Species_Carp",
        "Species_Cell", "Species_Chlorella vulgaris", "Species_Donax faba",
        "Species_Gobiocypris rarus", "Species_Grape (Vitis vinifera L.)", "Species_Grass Carp",
        "Species_Larimichthys crocea", "Species_Lemna minor",
        "Species_Mouse Sertoli", "Species_Mouse sertoli", "Species_Nostoc muscorum",
        "Species_Oreochromis niloticus", "Species_Rainbow trout", "Species_Rare minnow",
        "Species_Scenedesmus obliquus", "Species_Tetrahymena thermophila",
        "Species_Tomato (Solanum lycopersicum Mill)", "Species_Triticum aestivum"    ]

    soil_species = ["",
        "Species_Adult male rats", "Species_Adult rats", "Species_Albino rats",
        "Species_Black soil Eisenia foetida", "Species_Broilers", "Species_Earthworms",
        "Species_Eisenia fetida", "Species_Eisenis fetida", "Species_Eremias argus",
        "Species_Fluvo-aquic soil Eisenia foetida", "Species_Male Wistar rats",
        "Species_Male albino Wistar rats", "Species_Male mice", "Species_Male sprague",
        "Species_Pisum sativum", "Species_R-(-)Eisenia fetida", "Species_Rac Eisenia fetida",
        "Species_Red clay Eisenia foetida", "Species_S-(+)Eisenia fetida", "Species_Wistar rats"
    ]

    aquatic_tissues = ["",
        "Tissues_Anabaena laxa", "Tissues_Body tissue", "Tissues_Brain", "Tissues_Carp",
        "Tissues_Chlorella vulgaris", "Tissues_Embryo", "Tissues_F98",
        "Tissues_Foot tissue", "Tissues_Gill", "Tissues_Gill tissue", "Tissues_H9c2 cardiomyoblasts",
        "Tissues_HCT116", "Tissues_Hepatopancreas", "Tissues_Homogenate",
        "Tissues_Human erythrocytes", "Tissues_Larvae", "Tissues_Larval", "Tissues_Larval liver",
        "Tissues_Leaf", "Tissues_Lemna minor", "Tissues_Liver", "Tissues_Liver cells",
        "Tissues_Nostoc muscorum", "Tissues_Raw264.7 cells", "Tissues_Root",
        "Tissues_Scenedesmus obliquus", "Tissues_Spleen", "Tissues_TM4 cells",
    ]

    soil_tissues = ["",
        "Tissues_Bone", "Tissues_Brain", "Tissues_Erythrocyte", "Tissues_Heart tissue",
        "Tissues_Heart tissues", "Tissues_Kidney", "Tissues_Laval", "Tissues_Liver",
        "Tissues_Myocardial tissues", "Tissues_Shoots", "Tissues_Testis"
    ]

    # Function to update options based on medium selection
    def update_options(*args):
        medium = medium_var.get()
        
        # Update compound options
        compound_cb['values'] = aquatic_compounds if medium == "aquatic" else soil_compounds
        compound_cb.current(0)
        
        # Update species options
        species_cb['values'] = aquatic_species if medium == "aquatic" else soil_species
        species_cb.current(0)
        
        # Update tissue options
        tissue_cb['values'] = aquatic_tissues if medium == "aquatic" else soil_tissues
        tissue_cb.current(0)

    # Medium selection (Radio buttons: Aquatic / Soil)
    ttk.Label(left_frame, text="Medium:", style="Bold.TLabel").grid(
        row=1, column=0, sticky=tk.W, pady=(0, 15)
    )
    medium_var = tk.StringVar(value="aquatic")
    # Trace the medium variable to update options when changed
    medium_var.trace_add("write", update_options)
    
    ttk.Radiobutton(left_frame, text="Aquatic", variable=medium_var, value="aquatic").grid(
        row=1, column=1, sticky=tk.W, padx=10, pady=(0, 15)
    )
    ttk.Radiobutton(left_frame, text="Soil", variable=medium_var, value="soil").grid(
        row=1, column=2, sticky=tk.W, pady=(0, 15)
    )

    # Compound selection (Combobox)
    ttk.Label(left_frame, text="Compound:", style="Bold.TLabel").grid(
        row=2, column=0, sticky=tk.W, pady=(0, 15)
    )
    compound_var = tk.StringVar()
    compound_cb = ttk.Combobox(
        left_frame, textvariable=compound_var, values=aquatic_compounds, state="readonly"
    )
    compound_cb.grid(row=2, column=1, columnspan=2, sticky=tk.W + tk.E, pady=(0, 15))
    compound_cb.current(0)

    # Pollutant concentration input (with unit)
    ttk.Label(left_frame, text="Concentrations:", style="Bold.TLabel").grid(
        row=3, column=0, sticky=tk.W, pady=(0, 15)
    )
    conc_var = tk.StringVar()
    conc_entry = ttk.Entry(left_frame, textvariable=conc_var, width=20)
    conc_entry.grid(row=3, column=1, sticky=tk.W, pady=(0, 15))
    ttk.Label(left_frame, text="mg/kg or mg/L").grid(
        row=3, column=2, sticky=tk.W, padx=10, pady=(0, 15)
    )

    # Exposure time input (with unit)
    ttk.Label(left_frame, text="Exposure Time:", style="Bold.TLabel").grid(
        row=4, column=0, sticky=tk.W, pady=(0, 15)
    )
    time_var = tk.StringVar()
    time_entry = ttk.Entry(left_frame, textvariable=time_var, width=20)
    time_entry.grid(row=4, column=1, sticky=tk.W, pady=(0, 15))
    ttk.Label(left_frame, text="day(s)").grid(
        row=4, column=2, sticky=tk.W, padx=10, pady=(0, 15)
    )

    # Species selection (Combobox)
    ttk.Label(left_frame, text="Species:", style="Bold.TLabel").grid(
        row=5, column=0, sticky=tk.W, pady=(0, 15)
    )
    species_var = tk.StringVar()
    species_cb = ttk.Combobox(
        left_frame, textvariable=species_var, values=aquatic_species, state="readonly"
    )
    species_cb.grid(row=5, column=1, columnspan=2, sticky=tk.W + tk.E, pady=(0, 15))
    species_cb.current(0)

    # Tissue selection (Combobox)
    ttk.Label(left_frame, text="Tissue:", style="Bold.TLabel").grid(
        row=6, column=0, sticky=tk.W, pady=(0, 15)
    )
    tissue_var = tk.StringVar()
    tissue_cb = ttk.Combobox(
        left_frame, textvariable=tissue_var, values=aquatic_tissues, state="readonly"
    )
    tissue_cb.grid(row=6, column=1, columnspan=2, sticky=tk.W + tk.E, pady=(0, 15))
    tissue_cb.current(0)

    # Right frame: Result display and prediction function (inside bordered container)
    ttk.Label(right_frame, text="Prediction Results", style="Bold.TLabel").pack(
        pady=(0, 20)
    )

    mda_label = ttk.Label(right_frame, 
                          text="MDA: -\n(0: No Response, 1: Inhibition, 2: Stimulation)",
                          style="Bold.TLabel",
                          justify=tk.LEFT) 
    mda_label.pack(pady=15, anchor='w')  

    ros_label = ttk.Label(right_frame, 
                          text="ROS: -\n(0: No Response, 2: Stimulation)",
                          style="Bold.TLabel",
                          justify=tk.LEFT)  
    ros_label.pack(pady=15, anchor='w')  
    
    risk_label = ttk.Label(right_frame, text="Early Warning: -", style="Bold.TLabel", foreground="black")
    risk_label.pack(pady=30)

    def model_prediction():
        # Get input values
        medium = medium_var.get()
        compound = compound_var.get()
        conc_str = conc_var.get().strip()
        time_str = time_var.get().strip()
        species = species_var.get().strip()
        tissue = tissue_var.get().strip()

        # Input validation: Concentrations and Exposure Time cannot be empty
        if not conc_str or not time_str:
            messagebox.showerror("Input Error", "Concentrations and Exposure Time cannot be empty!")
            return

        try:
            conc = float(conc_str)
            time_val = float(time_str)
        except ValueError:
            messagebox.showerror("Input Error", "Concentrations and Exposure Time must be numeric!")
            return

        # Check at least one of species or tissue is selected
        if not species and not tissue:
            messagebox.showerror("Input Error", "At least one of Species or Tissue must be selected!")
            return

        # Define Excel file paths (in the program directory)
        current_dir = os.getcwd()
        if medium == "aquatic":
            mda_file = os.path.join(current_dir, "aquatic_MDA_train.xlsx")
            ros_file = os.path.join(current_dir, "aquatic_ROS_train.xlsx")
        else:
            mda_file = os.path.join(current_dir, "soil_MDA_train.xlsx")
            ros_file = os.path.join(current_dir, "soil_ROS_train.xlsx")

        # Check if Excel files exist
        for file_path in [mda_file, ros_file]:
            if not os.path.exists(file_path):
                messagebox.showerror(
                    "File Not Found",
                    f"Missing file: {os.path.basename(file_path)}\nPlease place it in the program directory."
                )
                return

        # Load training data and train models
        try:
            # Train MDA model
            mda_df = pd.read_excel(mda_file, sheet_name="Sheet1", engine="openpyxl")
            mda_X = mda_df.iloc[:, 2:]
            mda_y = mda_df.iloc[:, 1]
            feature_cols = mda_X.columns.tolist()

            if medium == "aquatic":
                mda_model = GradientBoostingClassifier(
                    n_estimators=423,
                    learning_rate=0.4454251663215166,
                    max_depth=9,
                    min_samples_split=17,
                    min_samples_leaf=2,
                    subsample=0.6821495488560924,
                    random_state=42
                )
            else:
                mda_model = HistGradientBoostingClassifier(
                    max_iter=77,
                    min_samples_leaf=1,
                    max_depth=8,
                    l2_regularization=0.0003912032355487126,
                    random_state=42
                )
            mda_model.fit(mda_X, mda_y)

            # Train ROS model
            ros_df = pd.read_excel(ros_file, sheet_name="Sheet1", engine="openpyxl")
            ros_X = ros_df.iloc[:, 2:]
            ros_y = ros_df.iloc[:, 1]

            if medium == "aquatic":
                ros_model = HistGradientBoostingClassifier(
                    max_iter=59,
                    min_samples_leaf=6,
                    max_depth=7,
                    l2_regularization=0.0004080785316419737,
                    random_state=42
                )
            else:
                ros_model = HistGradientBoostingClassifier(
                    max_iter=99,
                    min_samples_leaf=3,
                    max_depth=10,
                    l2_regularization=0.00022185096068270407,
                    random_state=42
                )
            ros_model.fit(ros_X, ros_y)

            # Construct input feature vector as DataFrame with column names
            input_data = {col: [0.0] for col in feature_cols}
            
            input_data["Concentrations"] = [conc]
            input_data["Time"] = [time_val]
            
            if compound in feature_cols:
                input_data[compound] = [1.0]

            if species and species in feature_cols:
                input_data[species] = [1.0]

            if tissue and tissue in feature_cols:
                input_data[tissue] = [1.0]

            # Create DataFrame with proper column names
            input_df = pd.DataFrame(input_data)

            # Model prediction and risk judgment
            mda_pred = mda_model.predict(input_df)[0]
            ros_pred = ros_model.predict(input_df)[0]

            if (mda_pred in [1, 2]) and (ros_pred == 2):
                risk_result = "Potential Risk"
                risk_color = "red"
            else:
                risk_result = "No Risk"
                risk_color = "green"

            # Update result display
            mda_label.config(text=f"MDA Prediction:\n{int(mda_pred)}\n(0: No Response, 1: Inhibition, 2: Stimulation)")
            ros_label.config(text=f"ROS Prediction:\n{int(ros_pred)}\n(0: No Response, 2: Stimulation)")
            risk_label.config(text=f"Early Warning Result:\n{risk_result}", foreground=risk_color)

            messagebox.showinfo("Completion", "Model prediction has been completed successfully!")
        
        except Exception as e:
            messagebox.showerror("Prediction Failed", f"Error: {str(e)}")

    # Configure button style
    style.configure("Accent.TButton", 
                   padding=10,
                   font=("Arial", 14), 
                   foreground="black",  
                   background="green")  

    style.map("Accent.TButton",
             background=[("active", "darkgreen"),  
                        ("pressed", "limegreen")])

    # Prediction button
    predict_btn = ttk.Button(
        right_frame, text="Model Prediction", command=model_prediction, style="Accent.TButton"
    )
    predict_btn.pack(pady=(0, 20), side=tk.TOP)


    root.mainloop()


if __name__ == "__main__":
    main()
