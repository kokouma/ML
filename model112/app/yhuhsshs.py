import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


data = {
    'Product_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Customer_Value': ['Premium', 'Standard', 'Economy', 'Premium', 'Economy', 'Standard', 'Premium', 'Economy', 'Standard', 'Premium'],
    'Sustainable_Materials': ['Upcycled', 'Biodegradable', 'Reusable', 'Repurposed', 'Eco-friendly', 'Innovative', 'Recyclable', 'Sustainable Wood', 'Bamboo', 'Cork'],
    'Renewable_Energy': ['Solar', 'Wind', 'Hydro', 'Solar', 'Hydro', 'Wind', 'Solar', 'Geothermal', 'Bioenergy', 'Tidal'],
    'Inclusive_Design': ['Accessible', 'Universal', 'Standard', 'Accessible', 'Standard', 'Universal', 'Accessible', 'Inclusive Language', 'User-friendly', 'Adaptable'],
    'Climate_Action_Keywords': ['Carbon Capture', 'Green Transportation', 'Agroecology', 'Renewable Energy', 'Energy Efficiency', 'Zero Emission', 'Circular Economy', 'Sustainable Packaging', 'Eco-friendly Construction', 'Waste Reduction'],
    'Quality_Education_Keywords': ['Online Learning Platforms', 'Educational Equality', 'Skill Development', 'Interactive Learning', 'Digital Literacy', 'Inclusive Curriculum', 'STEM Education', 'Global Citizenship Education', 'Distance Learning', 'Virtual Reality in Education'],
    'Decent_Work_Economic_Growth_Keywords': ['Fair Trade Practices', 'Remote Work Opportunities', 'Community Development', 'Sustainable Agriculture', 'Social Entrepreneurship', 'Financial Inclusion', 'Inclusive Hiring', 'Ethical Supply Chain', 'Labor Rights', 'Equal Pay'],
    'Industry_Innovation_Infrastructure_Keywords': ['Research and Development', 'Technological Innovation', 'Digital Transformation', 'Infrastructure Development', 'Smart Cities', 'Innovative Solutions', 'Clean Technology', 'Efficient Energy Infrastructure', 'Sustainable Transport', 'Green Infrastructure'],
    'Reduced_Inequalities_Keywords': ['Social Equity', 'Diversity and Inclusion', 'Anti-discrimination', 'Equal Opportunities', 'Inclusive Policies', 'Accessible Services', 'Affirmative Action', 'Community Engagement', 'Empowerment Programs', 'Gender Equality'],
    'Sustainable_Cities_Communities_Keywords': ['Urban Planning', 'Affordable Housing', 'Green Spaces', 'Public Transportation', 'Community Resilience', 'Cultural Diversity', 'Local Governance', 'Safe Cities', 'Smart Communities', 'Waste Management'],
    'Responsible_Consumption_Production_Keywords': ['Ethical Consumerism', 'Waste Reduction', 'Circular Economy', 'Product Lifecycle Management', 'Sustainable Sourcing', 'Fair Trade', 'Eco-labeling', 'Resource Efficiency', 'Green Manufacturing', 'Zero Waste'],
    'Climate_Action_Keywords': ['Renewable Energy', 'Energy Efficiency', 'Carbon Neutrality', 'Sustainable Agriculture', 'Forest Conservation', 'Climate Resilience', 'Climate Mitigation', 'Adaptation Strategies', 'Green Technologies', 'Climate Education'],
    'Life_Below_Water_Keywords': ['Marine Conservation', 'Ocean Sustainability', 'Sustainable Fisheries', 'Coral Reef Protection', 'Marine Biodiversity', 'Pollution Prevention', 'Plastic-Free Oceans', 'Eco-friendly Fishing', 'Ocean Cleanup', 'Water Quality Monitoring'],
    'Life_on_Land_Keywords': ['Biodiversity Conservation', 'Wildlife Protection', 'Ecosystem Restoration', 'Sustainable Agriculture', 'Deforestation Prevention', 'Anti-poaching Measures', 'Habitat Preservation', 'Reforestation Initiatives', 'Land Degradation Prevention', 'Indigenous Rights'],
    'Peace_Justice_Strong_Institutions_Keywords': ['Rule of Law', 'Access to Justice', 'Anti-corruption Measures', 'Human Rights Protection', 'Conflict Resolution', 'Equality Before the Law', 'Accountable Institutions', 'Transparency', 'Good Governance', 'Democratic Participation'],
    'Partnerships_for_the_Goals_Keywords': ['Global Collaboration', 'Public-Private Partnerships', 'Knowledge Sharing', 'Resource Mobilization', 'Sustainable Development Funding', 'Capacity Building', 'Inclusive Decision-Making', 'Stakeholder Engagement', 'Cross-Sectoral Cooperation', 'International Cooperation'],
    'Zero_Hunger_Keywords': ['Sustainable Agriculture', 'Food Security', 'Nutrition Education', 'Smallholder Farmers Support', 'Zero Waste Food Production', 'Community Gardens', 'Food Distribution Networks', 'Fair Trade Practices', 'Agricultural Innovation', 'Healthy Eating'],
    'Good_Health_Wellbeing_Keywords': ['Healthcare Access', 'Disease Prevention', 'Mental Health Support', 'Community Health Centers', 'Vaccination Programs', 'Sanitation Facilities', 'Clean Water Access', 'Healthy Lifestyle Promotion', 'Medical Research', 'Health Education'],
    'Gender_Equality_Keywords': ['Equal Pay', 'Gender Inclusive Policies', 'Women Empowerment', 'Gender-Based Violence Prevention', 'Equal Opportunities', 'Representation in Leadership', 'Work-Life Balance', 'Inclusive Work Environments', 'Gender-sensitive Education', 'Healthcare Equality'],
    'Clean_Water_Sanitation_Keywords': ['Water Conservation', 'Sanitation Infrastructure', 'Water Quality Monitoring', 'Community Water Management', 'Safe Drinking Water', 'Hygiene Education', 'Wastewater Treatment', 'Plastic-Free Waterways', 'Water Access in Rural Areas', 'Innovations in Water Technology'],
    'Affordable_Clean_Energy_Keywords': ['Renewable Energy Adoption', 'Energy Access for All', 'Energy Efficiency Programs', 'Clean Cooking Solutions', 'Off-Grid Energy Systems', 'Innovations in Solar Power', 'Green Building Technologies', 'Sustainable Transportation', 'Community-Based Energy Projects', 'Awareness on Energy Conservation'],
    'Life_on_Land_Keywords': ['Biodiversity Conservation', 'Wildlife Protection', 'Ecosystem Restoration', 'Sustainable Agriculture', 'Deforestation Prevention', 'Anti-poaching Measures', 'Habitat Preservation', 'Reforestation Initiatives', 'Land Degradation Prevention', 'Indigenous Rights'],
    'Peace_Justice_Strong_Institutions_Keywords': ['Rule of Law', 'Access to Justice', 'Anti-corruption Measures', 'Human Rights Protection', 'Conflict Resolution', 'Equality Before the Law', 'Accountable Institutions', 'Transparency', 'Good Governance', 'Democratic Participation'],
    'Partnerships_for_the_Goals_Keywords': ['Global Collaboration', 'Public-Private Partnerships', 'Knowledge Sharing', 'Resource Mobilization', 'Sustainable Development Funding', 'Capacity Building', 'Inclusive Decision-Making', 'Stakeholder Engagement', 'Cross-Sectoral Cooperation', 'International Cooperation'],
    'Sustainability_Score': [4.3, 3.9, 4.0, 4.2, 3.8, 4.1, 4.5, 4.2, 3.7, 4.4]
}

# Convert categorical variables to one-hot encoding
df = pd.DataFrame(data)

# Translate keywords to binary for all SDGs
df_binary = pd.get_dummies(df, columns=['Customer_Value', 'Sustainable_Materials', 'Renewable_Energy', 'Inclusive_Design',
                                        'Climate_Action_Keywords', 'Quality_Education_Keywords', 'Decent_Work_Economic_Growth_Keywords',
                                        'Industry_Innovation_Infrastructure_Keywords', 'Reduced_Inequalities_Keywords',
                                        'Sustainable_Cities_Communities_Keywords', 'Responsible_Consumption_Production_Keywords',
                                        'Climate_Action_Keywords', 'Life_Below_Water_Keywords', 'Life_on_Land_Keywords',
                                        'Peace_Justice_Strong_Institutions_Keywords', 'Partnerships_for_the_Goals_Keywords',
                                        'Zero_Hunger_Keywords', 'Good_Health_Wellbeing_Keywords', 'Gender_Equality_Keywords',
                                        'Clean_Water_Sanitation_Keywords', 'Affordable_Clean_Energy_Keywords', 'Life_on_Land_Keywords',
                                        'Peace_Justice_Strong_Institutions_Keywords', 'Partnerships_for_the_Goals_Keywords'])

# Define weights for each keyword
weights = {
    'Customer_Value_Premium': 0.1,
    'Customer_Value_Standard': 0.1,
    'Customer_Value_Economy': 0.1,
    'Sustainable_Materials_Upcycled': 0.1,
    'Sustainable_Materials_Biodegradable': 0.1,
    'Sustainable_Materials_Reusable': 0.1,
    'Renewable_Energy_Solar': 0.1,
    'Renewable_Energy_Wind': 0.1,
    'Renewable_Energy_Hydro': 0.1,
    'Inclusive_Design_Accessible': 0.1,
    'Inclusive_Design_Universal': 0.1,
    'Inclusive_Design_Standard': 0.1,
    'Climate_Action_Carbon_Capture': 0.1,
    'Climate_Action_Green_Transportation': 0.1,
    'Climate_Action_Agroecology': 0.1,
    'Quality_Education_Online_Learning_Platforms': 0.1,
    'Quality_Education_Educational_Equality': 0.1,
    'Quality_Education_Skill_Development': 0.1,
    'Decent_Work_Economic_Growth_Fair_Trade_Practices': 0.1,
    'Decent_Work_Economic_Growth_Remote_Work_Opportunities': 0.1,
    'Decent_Work_Economic_Growth_Community_Development': 0.1,
    'Industry_Innovation_Infrastructure_Research_and_Development': 0.1,
    'Industry_Innovation_Infrastructure_Technological_Innovation': 0.1,
    'Industry_Innovation_Infrastructure_Digital_Transformation': 0.1,
    'Reduced_Inequalities_Social_Equity': 0.1,
    'Reduced_Inequalities_Diversity_and_Inclusion': 0.1,
    'Reduced_Inequalities_Anti-discrimination': 0.1,
    'Sustainable_Cities_Communities_Urban_Planning': 0.1,
    'Sustainable_Cities_Communities_Affordable_Housing': 0.1,
    'Sustainable_Cities_Communities_Green_Spaces': 0.1,
    'Responsible_Consumption_Production_Ethical_Consumerism': 0.1,
    'Responsible_Consumption_Production_Waste_Reduction': 0.1,
    'Responsible_Consumption_Production_Circular_Economy': 0.1,
    'Climate_Action_Renewable_Energy': 0.1,
    'Climate_Action_Energy_Efficiency': 0.1,
    'Climate_Action_Carbon_Neutrality': 0.1,
    'Life_Below_Water_Marine_Conservation': 0.1,
    'Life_Below_Water_Ocean_Sustainability': 0.1,
    'Life_Below_Water_Sustainable_Fisheries': 0.1,
    'Life_on_Land_Biodiversity_Conservation': 0.1,
    'Life_on_Land_Wildlife_Protection': 0.1,
    'Life_on_Land_Ecosystem_Restoration': 0.1,
    'Peace_Justice_Strong_Institutions_Rule_of_Law': 0.1,
    'Peace_Justice_Strong_Institutions_Access_to_Justice': 0.1,
    'Peace_Justice_Strong_Institutions_Anti-corruption_Measures': 0.1,
    'Partnerships_for_the_Goals_Global_Collaboration': 0.1,
    'Partnerships_for_the_Goals_Public_Private_Partnerships': 0.1,
    'Partnerships_for_the_Goals_Knowledge_Sharing': 0.1,
    'Zero_Hunger_Sustainable_Agriculture': 0.1,
    'Zero_Hunger_Food_Security': 0.1,
    'Zero_Hunger_Nutrition_Education': 0.1,
    'Good_Health_Wellbeing_Healthcare_Access': 0.1,
    'Good_Health_Wellbeing_Disease_Prevention': 0.1,
    'Good_Health_Wellbeing_Mental_Health_Support': 0.1,
    'Gender_Equality_Equal_Pay': 0.1,
    'Gender_Equality_Gender_Inclusive_Policies': 0.1,
    'Gender_Equality_Women_Empowerment': 0.1,
    'Clean_Water_Sanitation_Water_Conservation': 0.1,
    'Clean_Water_Sanitation_Sanitation_Infrastructure': 0.1,
    'Clean_Water_Sanitation_Water_Quality_Monitoring': 0.1,
    'Affordable_Clean_Energy_Renewable_Energy': 0.1,
    'Affordable_Clean_Energy_Energy_Affordability': 0.1,
    'Affordable_Clean_Energy_Sustainable_Energy_Solutions': 0.1,
    'Life_on_Land_Wildlife_Protection': 0.1,
    'Life_on_Land_Ecosystem_Restoration': 0.1,
    'Peace_Justice_Strong_Institutions_Human_Rights_Protection': 0.1,
    'Peace_Justice_Strong_Institutions_Conflict_Resolution': 0.1,
    'Peace_Justice_Strong_Institutions_Equality_Before_the_Law': 0.1,
    'Partnerships_for_the_Goals_Resource_Mobilization': 0.1,
    'Partnerships_for_the_Goals_Sustainable_Development_Funding': 0.1,
    'Partnerships_for_the_Goals_Capacity_Building': 0.1,
    'Sustainability_Score': [4.3, 3.9, 4.0, 4.2, 3.8, 4.1, 4.5, 4.2, 3.7, 4.4]
}

def transform_text_data(df):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the 'user_input' column
    text_features = vectorizer.fit_transform(df['user_input'])

    # Convert the sparse matrix to a DataFrame
    text_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the text features with the original DataFrame
    df = pd.concat([df, text_df], axis=1)

    return df

# Calculate the weighted sum for each product
df['Weighted_Sum'] = df_binary.apply(lambda row: sum(row[key] * weights[key] for key in weights.keys() if key != 'Sustainability_Score' and key in row), axis=1)

def calculate_sustainability(df):
    # Transform text data
    df = transform_text_data(df)

    # Calculate the Sustainability Score
    df['Weighted_Sum'] = pd.to_numeric(df['Weighted_Sum'], errors='coerce')
    df['Sustainability_Score'] = pd.to_numeric(df['Sustainability_Score'], errors='coerce')

    # Calculate the Sustainability Score
    df['Calculated_Sustainability_Score'] = df['Weighted_Sum'].fillna(0) + df['Sustainability_Score'].fillna(0)

    # Display the calculated scores
    print("\nCalculated Sustainability Scores:")
    print(df[['Product_ID', 'Calculated_Sustainability_Score']])

# Train the linear regression model
def train_model(df_binary):

    # Features and target variable
    X = df_binary.drop(['Product_ID', 'Sustainability_Score'], axis=1)
    y = df_binary['Sustainability_Score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model