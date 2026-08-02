from predict_engine import run_prediction

print("\n=== Crypto Prediction AI ===\n")

asset = input("Asset (BTC/ETH/SOL): ").upper().strip()

direction_input = input("Direction (ABOVE/BELOW): ").upper().strip()

direction = 1 if direction_input == "ABOVE" else -1

target_price = float(input("Target price: "))

minutes = int(input("Prediction time in minutes: "))

result = run_prediction(asset, target_price, minutes, direction=direction)

print(f"\nUsing AI Model: {result['model_name']} ({result['direction']})")

if result["threshold_found"]:
    print(f"Decision Threshold: {result['decision_threshold']:.2f}")
else:
    print("No threshold file found. Using default threshold 0.50")

print(f"Using {result['data_source']} market data.")

print("\n" + "=" * 60)
print("CRYPTO PREDICTION REPORT")
print("=" * 60)

print(f"Asset              : {result['asset']}")
print(f"Direction          : {result['direction']}")
print(f"AI Model           : {result['model_name']}")
print(f"Decision Threshold : {result['decision_threshold']:.2f}")
print()

print(f"Current Price      : ${result['current_price']:,.2f}")
print(f"Target Price       : ${result['target_price']:,.2f}")
print(f"Required Move      : {result['required_return']*100:.2f}%")
print(f"Prediction Window  : {result['minutes']} minutes")

print()
print("=" * 60)
print("MARKET CONDITIONS")
print("=" * 60)

print(f"Trend              : {result['trend']}")
print(f"RSI                : {result['rsi']:.2f} ({result['rsi_text']})")
print(f"Volatility         : {result['volatility_status']}")
print(f"Current Volatility : {result['volatility']*100:.3f}%")

print()
print("=" * 60)
print("AI DECISION")
print("=" * 60)

print(f"Prediction         : {result['verdict']}")
print(f"Probability        : {result['probability']*100:.2f}%")
print(f"Confidence         : {result['confidence']}")
print(f"Risk Level         : {result['risk']}")

if result["outside_training"]:
    print()
    print("WARNING: Requested move is larger than most training examples.")
    print("Prediction reliability may decrease.")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

status = "LIKELY" if result["prediction_yes"] else "UNLIKELY"

direction_word = "ABOVE" if direction == 1 else "BELOW"

print(
    f"The AI predicts {result['asset']} is {status} to trade {direction_word} "
    f"${result['target_price']:,.2f} within {result['minutes']} minutes."
)
print(f"Probability: {result['probability']*100:.2f}%")

print()
print("=" * 60)
print("Prediction Complete")
print("=" * 60)