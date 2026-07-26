# PropNavigator — Price Prediction Architecture (Reference)

Streamlit ↔ FastAPI ↔ Model ↔ Docker, explained end-to-end. Written for the Price Prediction
module specifically (the other three modules — Analytics, Recommendation, Insight — read
precomputed data and don't go through FastAPI).

## The diagram

![PropNavigator architecture diagram](architecture_diagram.svg)

## The explanation

I built an end-to-end machine learning application rather than just a prediction model. I first
trained a LightGBM regression model and saved it as `best_model.joblib`. However, a trained model
by itself is just a file—it cannot receive requests from users or communicate with other
applications.

To allow users to interact with the model, I built a frontend using Streamlit. Streamlit provides
a web interface where users can enter property details and view the estimated price. However, I
intentionally did not put the machine learning logic inside Streamlit because I wanted the
prediction logic to be reusable by other applications in the future, such as a mobile app or
another website.

For that reason, I introduced a FastAPI backend. When the user submits the form, Streamlit sends
an HTTP POST request containing the property details as JSON to FastAPI. FastAPI acts as the
prediction service—it validates the request, performs the same preprocessing and feature
engineering used during training, prepares the input in the format expected by the model, and
calls the trained LightGBM model saved as `best_model.joblib`. The model returns the predicted
price, FastAPI wraps it into a JSON response, and Streamlit displays it to the user. By separating
the frontend from the backend, the machine learning logic exists in one place and can be reused by
any client instead of being duplicated across multiple applications.

Finally, for deployment, I packaged the backend—including FastAPI, the trained model, Python
dependencies, and required runtime libraries—inside a Docker container. I used Docker because
applications often behave differently across machines due to differences in operating systems,
Python versions, or installed libraries. Docker packages the entire runtime environment, ensuring
that the backend runs consistently across development, testing, and production environments
without environment-specific issues.

## The three "why" statements to remember for interviews

- **Why Streamlit?**
  To provide a simple web interface so users can interact with the model without writing code or
  making API calls.
- **Why FastAPI?**
  To separate the machine learning logic from the user interface and expose the model as a
  reusable prediction service that any client can consume.
- **Why Docker?**
  To package the backend and all of its dependencies into a consistent environment so the
  application runs the same way on any machine or cloud server.
