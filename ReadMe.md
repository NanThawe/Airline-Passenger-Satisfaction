# Problem Description

The airline industry faces an ongoing challenge in understanding and enhancing passenger satisfaction. The quality of the air travel experience is influenced by various factors such as in-flight services, seat comfort, on-time performance, and overall customer service. Airlines seek to optimize these elements to improve passenger satisfaction, loyalty, and, consequently, their competitiveness in the market. To address this, predicting airline passenger satisfaction becomes crucial for airlines to proactively identify potential issues and implement targeted improvements

This is a valuable tool for airlines to enhance the overall travel experience, improve operational efficiency, retain customers, and stay competitive in the dynamic aviation industry.

## How this solution will be used? 

This prediction model can be applied in four ways to benefit both the airline and passengers: 

**Service Enhancement**:
By predicting passenger satisfaction, airlines can identify specific areas that need improvement. This insight allows them to focus resources on enhancing services, such as in-flight entertainment, meal quality, or cabin comfort, leading to an overall better travel experience.

**Operational Efficiency**:
Understanding factors influencing satisfaction helps airlines optimize their operations. Predictions can be used to anticipate passenger preferences, streamline boarding processes, and allocate resources effectively, leading to improved efficiency and reduced operational costs.

**Customer Retention**:
Airlines can use satisfaction predictions to implement proactive measures to retain customers. Addressing potential issues before they escalate can significantly impact customer loyalty, encouraging repeat business and positive word-of-mouth.

**Marketing Strategies**:
Predictive models can inform targeted marketing strategies. Airlines can tailor promotional campaigns based on the factors most influential in passenger satisfaction, attracting new customers and differentiating themselves in a competitive market.

## Data description

**Link to Dataset** : https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data 

**Gender**: Gender of the passengers (Female, Male)

**Customer Type**: The customer type (Loyal customer, disloyal customer)

**Age**: The actual age of the passengers

**Type of Travel**: Purpose of the flight of the passengers (Personal Travel, Business Travel)

**Class**: Travel class in the plane of the passengers (Business, Eco, Eco Plus)

**Flight distance**: The flight distance of this journey

**Inflight wifi service**: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)

**Departure/Arrival time convenient**: Satisfaction level of Departure/Arrival time convenient

**Ease of Online booking**: Satisfaction level of online booking

**Gate location**: Satisfaction level of Gate location

**Food and drink**: Satisfaction level of Food and drink

**Online boarding**: Satisfaction level of online boarding

**Seat comfort**: Satisfaction level of Seat comfort

**Inflight entertainment**: Satisfaction level of inflight entertainment

**On-board service**: Satisfaction level of On-board service

**Leg room service**: Satisfaction level of Leg room service

**Baggage handling**: Satisfaction level of baggage handling

**Check-in service**: Satisfaction level of Check-in service

**Inflight service**: Satisfaction level of inflight service

**Cleanliness**: Satisfaction level of Cleanliness

**Departure Delay in Minutes**: Minutes delayed when departure

**Arrival Delay in Minutes**: Minutes delayed when Arrival

**Satisfaction: Airline**: satisfaction level(Satisfaction, neutral or dissatisfaction)

# Steamlit App
[Airline Passenger Satisfaction Steamlit App link](https://airline-passenger-satisfaction.streamlit.app/)
![Screenshot of the Steamlit app](https://github.com/NanThawe/Airline-Passenger-Satisfaction/blob/main/steamlit_app.png)

# Docker 
1. Build the Docker image
   ```
   docker build -t airline-steamlit .

   ```
2. Run the Docker container
   ```
   docker run -p 8501:8501 airline-steamlit

   ```

3. Open app on localhost
   `http://0.0.0.0:8501`
   
   

