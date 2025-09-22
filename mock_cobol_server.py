from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# サンプルの製品データ
PRODUCTS = {
    "P001": {"product_name": "産業用ロボットアーム", "category": "machinery"},
    "P002": {"product_name": "自動車エンジン", "category": "automotive"},
    "P003": {"product_name": "スマートフォン", "category": "electronics"},
    "P004": {"product_name": "ノートパソコン", "category": "electronics"},
    "P005": {"product_name": "製造装置", "category": "machinery"}
}

@app.route('/get_product_info', methods=['GET'])
def get_product_info():
    product_id = request.args.get('id')
    
    if not product_id:
        return jsonify({'error': '製品IDが指定されていません'}), 400
    
    product = PRODUCTS.get(product_id)
    
    if not product:
        return jsonify({'error': '製品が見つかりません'}), 404
    
    return jsonify({
        'product_id': product_id,
        'product_name': product['product_name'],
        'category': product['category'],
        'status': 'success'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'cobol-mock-server',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'COBOL API Mock Server',
        'version': '1.0.0',
        'products': list(PRODUCTS.keys()),
        'endpoints': [
            '/get_product_info?id=P001 (GET) - 製品情報取得',
            '/health (GET) - ヘルスチェック'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))
