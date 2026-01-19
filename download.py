from gliner import GLiNER

# Cela va télécharger le modèle et le tokenizer
print("Téléchargement en cours...")
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Cela va sauvegarder TOUT ce qu'il faut (config, poids, tokenizer) dans un dossier local
print("Sauvegarde locale...")
model.save_pretrained("./models/gliner_local")
print("Terminé ! Copiez le dossier 'models' sur votre machine offline.")