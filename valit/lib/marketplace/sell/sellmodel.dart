// class Product {
//   final String productname;
//   final double productprice;
//   final String productuploaddate;
//   final String productimage;
//   final String hotellocation;
//   final String productId;

//   Product(
//       {required this.productname,
//       required this.productprice,
//       required this.productuploaddate,
//       required this.productimage,
//       required this.hotellocation,
//       required this.productId});
// }

class Product {
  final String productname;
  final double productprice;
  final String productuploaddate;
  final String productimage;
  final String productlocation;
  // final String userId;
  // final int phonenumber;
  final String productId;

  Product(
      {required this.productname,
      required this.productprice,
      required this.productuploaddate,
      // required this.userId,
      required this.productimage,
      required this.productlocation,
      // required this.phonenumber,
      required this.productId});
}
