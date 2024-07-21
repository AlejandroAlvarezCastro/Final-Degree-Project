import yaml
import os

class ConfigGenerator:
    def __init__(self, models, epochs, frozen_layers):
        """
        Initializes the ConfigGenerator with models, epochs, and frozen layers.

        Parameters:
            models (list): List of model names or identifiers.
            epochs (list): List of epoch values.
            frozen_layers (list): List of frozen layer counts.
        """
        self.configurations = {}
        self.models = models
        self.epochs = epochs
        self.frozen_layers = frozen_layers

    def generate_configurations(self):
        """
        Generates configurations based on the models, epochs, and frozen layers.

        This method creates a dictionary of configurations where each key is a
        unique configuration name and the value is a dictionary containing the
        model, epochs, frozen layers, tags, and metadata.
        """
        for model in self.models:
            for layers in self.frozen_layers:
                for epoch in self.epochs:
                    config_name = f"{model}_{layers}_{epoch}"
                    self.configurations[config_name] = {
                        'model': model,
                        'epochs': epoch,
                        'frozen_layers': layers,
                        'tags': [str(model), f'{epoch}_epochs', f'freeze_{layers}'],
                        'metadata': {
                            'model': model,
                            'epochs': epoch,
                            'frozen_layers': layers,
                            config_name: f"{model}_{layers}_{epoch}"
                        }
                    }

    def save_to_yaml(self, file_path):
        """
        Saves the generated configurations to a YAML file.

        Parameters:
            file_path (str): The path where the YAML file will be saved.
        """
        data = {'configurations': self.configurations}
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

# Ejemplo de uso
if __name__ == "__main__":
    # Modelos Yolo8
    directory = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/cls-models'
    pt_files = [file for file in os.listdir(directory) if file.endswith('.pt')]

    # Parámetros
    epochs = [50, 100, 300]
    layers = [0, 3, 5, 8, 9]

    # Crear una instancia del generador de configuraciones
    generator = ConfigGenerator(pt_files, epochs, layers)

    # Generar configuraciones
    generator.generate_configurations()

    # Guardar en un archivo YAML
    generator.save_to_yaml('/home/aacastro/Alejandro/DQ_ACA_2024/B_def/configurations.yaml')

    # Imprimir la cantidad de configuraciones generadas
    print(f"Se han generado {len(generator.configurations)} configuraciones.")
